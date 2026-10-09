"""
Collect and report health information for layered-product image groups.

Collection and report formatting use shared image-health helpers while this
module handles product-specific variants and an aggregated Slack report.
"""

import asyncio
from dataclasses import dataclass

import click
from artcommonlib import exectools
from artcommonlib.variants import BuildVariant, get_build_variant_for_product
from doozerlib.constants import ART_BUILD_FAILURES_URL

from pyartcd import util
from pyartcd.cli import cli, click_coroutine, pass_runtime
from pyartcd.constants import OCP_BUILD_DATA_URL
from pyartcd.pipelines.image_health_common import (
    build_group_param,
    build_history_url,
    build_logs_url,
    filter_failure_concerns,
    filter_failure_map,
    filter_image_list,
    filter_image_names,
    format_build_concern_line,
    format_counter_failure_section,
    format_rebase_failure_section,
    get_counter_failures,
    get_rebase_failures,
    get_valid_images_from_group,
    run_images_health,
    slack_url_text,
)
from pyartcd.runtime import Runtime
from pyartcd.slack import SlackClient


def _parse_groups(groups: str) -> list[str]:
    """
    Parse a comma-separated layered-product group list.

    Args:
        groups: Comma-separated group names.

    Returns:
        Group names with whitespace removed and duplicates removed in order.

    Raises:
        ValueError: If no group name is provided.
    """
    parsed_groups = list(dict.fromkeys(group.strip() for group in groups.split(",") if group.strip()))
    if not parsed_groups:
        raise ValueError("At least one layered-product group is required")
    return parsed_groups


@dataclass
class LayeredProductHealthReport:
    """
    Store health information collected for one layered-product group.
    """

    group: str
    product: str
    variant: BuildVariant | None
    build_concerns: list[dict]
    build_failures: dict[str, dict]
    its_failures: dict[str, dict]
    release_failures: dict[str, dict]
    rebase_failures: dict[str, dict]
    error: str | None = None


class LayeredProductsImageHealthPipeline:
    """
    Collect and report health data for layered-product image groups.
    """

    def __init__(
        self,
        runtime: Runtime,
        groups: str,
        data_path: str,
        data_gitref: str,
        assembly: str,
        image_list: str,
    ) -> None:
        """
        Initialize the layered-product health pipeline.

        Args:
            runtime: The pyartcd runtime.
            groups: Comma-separated layered-product groups.
            data_path: ocp-build-data repository path.
            data_gitref: Optional ocp-build-data Git reference.
            assembly: Assembly to inspect.
            image_list: Optional comma-separated image filter.
        """
        self.runtime = runtime
        self.groups = _parse_groups(groups)
        self.data_path = data_path
        self.data_gitref = data_gitref
        self.assembly = assembly
        self.image_list = [image.strip() for image in image_list.split(",") if image.strip()]
        self.slack_client = self.runtime.new_slack_client()
        self._doozer_working = self.runtime.working_dir / "doozer_working"

    async def run(self) -> None:
        """
        Collect all configured groups and publish one aggregated report.
        """
        reports = await asyncio.gather(*(self._collect_group_safely(group) for group in self.groups))
        reports = sorted(reports, key=lambda report: report.group)
        await self._notify_slack(reports)

        failed_groups = [report.group for report in reports if report.error]
        if failed_groups:
            raise RuntimeError(f"Layered product health collection failed for: {', '.join(failed_groups)}")

    async def _collect_group_safely(self, group: str) -> LayeredProductHealthReport:
        """
        Collect one group while preserving an error report for aggregation.

        Args:
            group: Layered-product group name.

        Returns:
            A successful group report or a report containing the collection error.
        """
        try:
            return await self._collect_group(group)
        except Exception as error:
            self.runtime.logger.exception("Failed to collect layered product health for %s", group)
            return LayeredProductHealthReport(
                group=group,
                product="unknown",
                variant=None,
                build_concerns=[],
                build_failures={},
                its_failures={},
                release_failures={},
                rebase_failures={},
                error=str(error),
            )

    async def _collect_group(self, group: str) -> LayeredProductHealthReport:
        """
        Collect variant-aware health data for one layered-product group.

        Args:
            group: Layered-product group name.

        Returns:
            Health data collected for the group.
        """
        group_config = await util.load_group_config(
            group=group,
            assembly=self.assembly,
            doozer_data_path=self.data_path,
            doozer_data_gitref=self.data_gitref,
        )
        product = group_config.get("product")
        if not product:
            raise ValueError(f"No product found in group config for {group}")
        variant = get_build_variant_for_product(product)

        build_failures, its_failures, release_failures, rebase_failures = await asyncio.gather(
            get_counter_failures(
                "build-failure",
                group,
                self.runtime.logger,
                build_variant=variant.value,
            ),
            get_counter_failures(
                "ec-failure",
                group,
                self.runtime.logger,
                build_variant=variant.value,
            ),
            get_counter_failures(
                "release-failure",
                group,
                self.runtime.logger,
                build_variant=variant.value,
            ),
            get_rebase_failures(
                group,
                self.runtime.logger,
                build_systems=["konflux"],
                build_variant=variant.value,
            ),
        )

        failure_images = set(build_failures) | set(its_failures) | set(release_failures)
        valid_images = set()
        if failure_images or rebase_failures:
            valid_images = await self._get_valid_images(group, variant)
            build_failures = filter_failure_map(
                build_failures, valid_images, group, self.runtime.logger, "build failure"
            )
            its_failures = filter_failure_map(its_failures, valid_images, group, self.runtime.logger, "ec failure")
            release_failures = filter_failure_map(
                release_failures, valid_images, group, self.runtime.logger, "release failure"
            )
            rebase_failures = filter_failure_map(
                rebase_failures, valid_images, group, self.runtime.logger, "rebase failure"
            )

        failure_images = filter_image_list(
            set(build_failures) | set(its_failures) | set(release_failures), set(self.image_list)
        )

        build_concerns = await self._get_build_concerns(group, variant, failure_images, valid_images)
        return LayeredProductHealthReport(
            group=group,
            product=product,
            variant=variant,
            build_concerns=build_concerns,
            build_failures=build_failures,
            its_failures=its_failures,
            release_failures=release_failures,
            rebase_failures=rebase_failures,
        )

    async def _get_build_concerns(
        self,
        group: str,
        variant: BuildVariant,
        image_names: set[str],
        valid_images: set[str] | None = None,
    ) -> list[dict]:
        """
        Return build concerns for affected images.

        Args:
            group: Layered-product group name.
            variant: Product-specific build variant.
            image_names: Redis-reported image names to inspect.

        Returns:
            Doozer health concerns for the group.
        """
        if not image_names:
            return []

        if valid_images is None:
            valid_images = await self._get_valid_images(group, variant)
        filtered_images = filter_image_names(image_names, valid_images, group, self.runtime.logger)
        if not filtered_images:
            return []

        working_dir = self._doozer_working / group
        command = [
            "doozer",
            f"--working-dir={working_dir}",
            f"--data-path={self.data_path}",
            build_group_param(group, self.data_gitref),
            f"--assembly={self.assembly}",
            "--build-system=konflux",
            f"--variant={variant.value}",
        ]
        report, _ = await run_images_health(command, filtered_images, command_runner=exectools.cmd_gather_async)
        return report

    async def _get_valid_images(self, group: str, variant: BuildVariant) -> set[str]:
        """
        Return image names currently defined for a layered-product group.

        Args:
            group: Layered-product group name.
            variant: Product-specific build variant.

        Returns:
            Image names available in the selected group and variant.
        """
        return await get_valid_images_from_group(
            group=group,
            assembly=self.assembly,
            build_system="konflux",
            working_dir=self._doozer_working / group,
            data_path=self.data_path,
            data_gitref=self.data_gitref,
            variant=variant.value,
        )

    def _build_summary_message(self, reports: list[LayeredProductHealthReport]) -> str:
        """
        Build the aggregated Slack parent message.

        Args:
            reports: Group health reports sorted for display.

        Returns:
            Slack-formatted summary text.
        """
        message_parts = [":alert: Layered product image health report:"]
        for report in reports:
            if report.error:
                message_parts.append(f"- `{report.group}`: :warning: incomplete ({report.error})")
                continue

            failure_parts = []
            build_concerns = self._get_failure_concerns(report.build_concerns)
            if build_concerns:
                failure_parts.append(f"{len(build_concerns)} build failure(s)")
            if report.its_failures:
                failure_parts.append(f"{len(report.its_failures)} ITS failure(s)")
            if report.release_failures:
                failure_parts.append(f"{len(report.release_failures)} release failure(s)")
            if report.rebase_failures:
                failure_parts.append(f"{len(report.rebase_failures)} rebase failure(s)")

            if failure_parts:
                summary = ", ".join(failure_parts)
                message_parts.append(f"- `{report.group}` ({report.product}): {summary}")
            else:
                message_parts.append(f"- `{report.group}` ({report.product}): :white_check_mark: healthy")

        message_parts.append(f"\nFor details, see <{ART_BUILD_FAILURES_URL}|ART Build Failures Dashboard>.")
        return "\n".join(message_parts)

    def _build_group_message(self, report: LayeredProductHealthReport) -> str:
        """
        Build one detailed Slack thread section for a product group.

        Args:
            report: Group health report to format.

        Returns:
            Slack-formatted group details.
        """
        variant = report.variant.value if report.variant else "unknown"
        header = f"*{report.product}* (`{report.group}`, variant `{variant}`)"
        if report.error:
            return f"{header}\n:warning: Report incomplete: {report.error}"

        sections = [header]
        build_concerns = self._get_failure_concerns(report.build_concerns)
        if build_concerns:
            lines = [f"*Build Failures ({len(build_concerns)}):*"]
            lines.extend(self._format_build_concern(concern, report.group) for concern in build_concerns)
            sections.append("\n".join(lines))
        if report.its_failures:
            sections.append(self._format_counter_section("ITS Verification Failures", report.its_failures))
        if report.release_failures:
            sections.append(self._format_counter_section("Release to Authz Failures", report.release_failures))
        if report.rebase_failures:
            sections.append(
                format_rebase_failure_section(
                    report.rebase_failures,
                    self._url_text,
                    "Last failure",
                    include_pipeline_url=True,
                )
            )
        if len(sections) == 1:
            sections.append(":white_check_mark: Healthy")
        return "\n\n".join(sections)

    async def _notify_slack(self, reports: list[LayeredProductHealthReport]) -> None:
        """
        Send the aggregated parent message and per-group thread sections.

        Args:
            reports: Group health reports sorted for display.
        """
        self.slack_client.bind_channel(SlackClient.DEFAULT_CHANNEL_LAYERED_OPERATORS)
        response = await self.slack_client.say(
            self._build_summary_message(reports),
            link_build_url=False,
            unfurl_links=False,
            unfurl_media=False,
        )
        for report in reports:
            await self.slack_client.say(
                self._build_group_message(report),
                thread_ts=response["ts"],
                unfurl_links=False,
                unfurl_media=False,
            )

    @staticmethod
    def _get_failure_concerns(concerns: list[dict]) -> list[dict]:
        """
        Exclude successful-build entries from a health report.

        Args:
            concerns: Raw Doozer health concerns.

        Returns:
            Concerns representing a failed or never-built image.
        """
        return filter_failure_concerns(concerns)

    def _format_build_concern(self, concern: dict, group: str) -> str:
        """
        Format one build concern with history and optional log links.

        Args:
            concern: Doozer health concern.
            group: Layered-product group associated with the concern.

        Returns:
            Slack-formatted build concern line.
        """
        return format_build_concern_line(
            concern,
            lambda: self._build_history_url(group, concern["image_name"]),
            lambda: self._build_logs_url(concern),
            self._url_text,
        )

    @staticmethod
    def _format_counter_section(title: str, failures: dict[str, dict]) -> str:
        """
        Format a Redis failure-counter section.

        Args:
            title: Section title.
            failures: Image failure data keyed by image name.

        Returns:
            Slack-formatted counter section.
        """
        return format_counter_failure_section(title, failures, LayeredProductsImageHealthPipeline._url_text)

    def _build_history_url(self, group: str, image_name: str) -> str:
        """
        Build an ART build-history URL for an image and group.

        Args:
            group: Layered-product group name.
            image_name: Image name.

        Returns:
            ART build-history search URL.
        """
        return build_history_url(image_name, group, self.assembly)

    @staticmethod
    def _build_logs_url(concern: dict) -> str:
        """
        Build a logs URL when the concern contains complete failure metadata.

        Args:
            concern: Doozer health concern.

        Returns:
            ART logs URL or an empty string when metadata is incomplete.
        """
        return build_logs_url(concern, encode_after=True, strict=False)

    @staticmethod
    def _url_text(url: str, text: str) -> str:
        """
        Format a URL as Slack link text.

        Args:
            url: URL to link.
            text: Display text.

        Returns:
            Slack link markup.
        """
        return slack_url_text(url, text)


@cli.command("layered-products-image-health")
@click.option("--groups", required=True, help="Comma-separated layered-product groups to scan")
@click.option("--assembly", required=False, default="stream", help="Assembly to scan for")
@click.option(
    "--data-path",
    required=False,
    default=OCP_BUILD_DATA_URL,
    help="ocp-build-data fork to use (e.g. assembly definition in your own fork)",
)
@click.option("--data-gitref", required=False, default="", help="Doozer data path git [branch / tag / sha] to use")
@click.option("--image-list", required=False, default="", help="Comma-separated list of images to scan")
@pass_runtime
@click_coroutine
async def layered_products_image_health(
    runtime: Runtime,
    groups: str,
    assembly: str,
    data_path: str,
    data_gitref: str,
    image_list: str,
) -> None:
    """
    Collect and report health for comma-separated layered-product groups.
    """
    await LayeredProductsImageHealthPipeline(
        runtime=runtime,
        groups=groups,
        data_path=data_path,
        data_gitref=data_gitref,
        assembly=assembly,
        image_list=image_list,
    ).run()
