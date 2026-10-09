"""
Shared collection and formatting helpers for image-health pipelines.

The OCP, OKD, and layered-product pipelines use these helpers for Redis
failure data, image metadata filtering, Doozer health concerns, and common
Slack report details. Pipeline-specific orchestration and destinations stay
in their respective modules.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from artcommonlib import exectools
from doozerlib.cli.images_health import DELTA_DAYS, LIMIT_BUILD_RESULTS, ConcernCode
from doozerlib.constants import ART_BUILD_HISTORY_URL

from pyartcd import util
from pyartcd.runtime import Runtime


def build_group_param(group: str, data_gitref: str) -> str:
    """
    Build a Doozer group argument with an optional ocp-build-data reference.

    Args:
        group: Doozer group name.
        data_gitref: Optional Git reference for ocp-build-data.

    Returns:
        A group argument suitable for a Doozer command.
    """
    if data_gitref:
        group += f"@{data_gitref}"
    return f"--group={group}"


async def get_valid_images_from_doozer(
    runtime: Runtime,
    group: str,
    working_dir: str | Path,
    data_path: str,
    data_gitref: str,
    variant: str | None = None,
    image_list_options: tuple[str, ...] = ("--short", "{distgit_key}"),
) -> set[str]:
    """
    Get valid image names with Doozer's images:print command.

    Args:
        runtime: pyartcd runtime used for logging.
        group: Doozer group name.
        working_dir: Doozer working directory.
        data_path: ocp-build-data repository path.
        data_gitref: Optional Git reference for ocp-build-data.
        variant: Optional Doozer build variant.
        image_list_options: Options passed to images:print after the command.

    Returns:
        Image distgit keys. Returns an empty set when the Doozer command fails.
    """
    command = [
        "doozer",
        f"--working-dir={working_dir}",
        f"--data-path={data_path}",
    ]
    if variant:
        command.append(f"--variant={variant}")
    command.extend(
        [
            build_group_param(group, data_gitref),
            "images:print",
            *image_list_options,
        ]
    )

    try:
        _, output, _ = await exectools.cmd_gather_async(command, stderr=None)
        images = {line.strip() for line in output.strip().splitlines() if line.strip()}
        runtime.logger.info("Found %d valid images for %s", len(images), group)
        return images
    except Exception as error:
        runtime.logger.warning("Failed to fetch valid images for %s: %s. Proceeding without filtering.", group, error)
        return set()


async def get_valid_images_from_group(
    group: str,
    assembly: str,
    build_system: str,
    working_dir: Path,
    data_path: str,
    data_gitref: str,
    variant: str | None = None,
) -> set[str]:
    """
    Get valid image names through pyartcd's group-image helper.

    Args:
        group: Doozer group name.
        assembly: Assembly to inspect.
        build_system: Build system to inspect.
        working_dir: Doozer working directory.
        data_path: ocp-build-data repository path.
        data_gitref: Optional Git reference for ocp-build-data.
        variant: Optional Doozer build variant.

    Returns:
        Image distgit keys defined for the group and assembly.
    """
    images = await util.get_group_images(
        group=group,
        assembly=assembly,
        build_system=build_system,
        working_dir=working_dir,
        doozer_data_path=data_path,
        doozer_data_gitref=data_gitref,
        variant=variant,
    )
    return set(images)


def filter_image_names(
    image_names: set[str],
    valid_images: set[str],
    group: str,
    logger: logging.Logger,
) -> set[str]:
    """
    Keep only metadata-defined image names and log stale Redis entries.

    Args:
        image_names: Image names reported by Redis.
        valid_images: Image names currently defined in metadata.
        group: Group used for the metadata lookup.
        logger: Logger for stale-image warnings.

    Returns:
        Image names present in both Redis and current metadata.
    """
    filtered_images = image_names & valid_images
    skipped_images = image_names - valid_images
    if skipped_images:
        logger.warning(
            "Filtered out %d image(s) from Redis that do not exist in %s metadata: %s",
            len(skipped_images),
            group,
            ", ".join(sorted(skipped_images)),
        )
    return filtered_images


def filter_image_list(image_names: set[str], image_list: set[str]) -> set[str]:
    """
    Apply an optional explicit image filter to a set of Redis image names.

    Args:
        image_names: Image names reported by Redis.
        image_list: Explicitly selected image names, or an empty set.

    Returns:
        All Redis image names when no explicit filter is set, otherwise their intersection.
    """
    return image_names & image_list if image_list else image_names


def filter_failure_map(
    failures: dict[str, dict],
    valid_images: set[str],
    group: str,
    logger: logging.Logger,
    failure_type: str,
) -> dict[str, dict]:
    """
    Keep Redis failure entries for images currently present in metadata.

    Args:
        failures: Redis failures keyed by image name.
        valid_images: Image names currently defined in metadata.
        group: Group used for the metadata lookup.
        logger: Logger for stale-image warnings.
        failure_type: Label used in the warning, such as "rebase failure".

    Returns:
        Failure entries whose image names are present in current metadata.
    """
    filtered_failures = {image: info for image, info in failures.items() if image in valid_images}
    skipped_images = failures.keys() - valid_images
    if skipped_images:
        logger.warning(
            "Filtered out %d %s(s) from Redis that do not exist in %s metadata: %s",
            len(skipped_images),
            failure_type,
            group,
            ", ".join(sorted(skipped_images)),
        )
    return filtered_failures


async def get_counter_failures(
    counter_type: str,
    group: str,
    logger: logging.Logger,
    build_system: str = "konflux",
    build_variant: str | None = None,
) -> dict[str, dict]:
    """
    Fetch failure entries from one Redis counter.

    Args:
        counter_type: Redis counter type, such as "build-failure".
        group: Redis group name.
        logger: Logger passed to the Redis helper.
        build_system: Build system recorded in Redis.
        build_variant: Optional Redis build-variant filter.

    Returns:
        Failure entries keyed by image name.
    """
    counter_options: dict[str, object] = {"group": group, "logger": logger}
    if build_system != "konflux":
        counter_options["build_system"] = build_system
    if build_variant:
        counter_options["build_variant"] = build_variant
    return await util.get_counter_failures(counter_type, **counter_options)


async def get_rebase_failures(
    group: str,
    logger: logging.Logger,
    build_systems: list[str],
    build_variant: str | None = None,
) -> dict[str, dict]:
    """
    Fetch rebase failure entries from Redis.

    Args:
        group: Redis group name.
        logger: Logger passed to the Redis helper.
        build_systems: Build systems to include in the query.
        build_variant: Optional Redis build-variant filter.

    Returns:
        Rebase failures keyed by image name.
    """
    rebase_options: dict[str, object] = {
        "group": group,
        "branches": ["rebase-failure"],
        "build_systems": build_systems,
        "logger": logger,
    }
    if build_variant:
        rebase_options["build_variant"] = build_variant
    return await util.get_rebase_failures(**rebase_options)


async def get_filtered_rebase_failures(
    group: str,
    valid_images: set[str],
    logger: logging.Logger,
    build_systems: list[str],
    build_variant: str | None = None,
) -> dict[str, dict]:
    """
    Fetch and metadata-filter Redis rebase failures.

    Args:
        group: Redis group name.
        valid_images: Image names currently defined in metadata.
        logger: Logger for Redis access and stale-image warnings.
        build_systems: Build systems to include in the query.
        build_variant: Optional Redis build-variant filter.

    Returns:
        Rebase failures for images present in current metadata.
    """
    failures = await get_rebase_failures(group, logger, build_systems, build_variant)
    return filter_failure_map(failures, valid_images, group, logger, "rebase failure")


async def run_images_health(
    command: list[str],
    image_names: set[str],
    after_command: tuple[str, ...] = (),
    command_runner: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
) -> tuple[list[dict], str]:
    """
    Run Doozer images:health for a sorted set of images.

    Args:
        command: Doozer base command, before the image selector.
        image_names: Images to query.
        after_command: Optional arguments placed after images:health.
        command_runner: Optional async command runner, mainly for pipeline-level patching.

    Returns:
        Parsed health concerns and raw stdout.
    """
    full_command = [
        *command,
        f"--images={','.join(sorted(image_names))}",
        "images:health",
        *after_command,
    ]
    runner = command_runner or exectools.cmd_gather_async
    _, output, _ = await runner(full_command, stderr=None)
    return (json.loads(output.strip()) if output.strip() else []), output


def build_history_url(
    image_name: str,
    group: str,
    assembly: str,
    build_system: str = "konflux",
) -> str:
    """
    Build an ART build-history URL for an image and group.

    Args:
        image_name: Image distgit key.
        group: Group to show in build history.
        assembly: Assembly to show in build history.
        build_system: Build system to show in build history.

    Returns:
        ART build-history search URL.
    """
    start_date = (datetime.now(timezone.utc) - timedelta(days=DELTA_DAYS)).strftime("%Y-%m-%d")
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"{ART_BUILD_HISTORY_URL}/?name=^{image_name}$&group={group}&assembly={assembly}"
        f"&engine={build_system}&dateRange={start_date}+to+{end_date}&outcome=Success&outcome=Failure"
    )


def build_logs_url(concern: dict, encode_after: bool = False, strict: bool = True) -> str:
    """
    Build an ART build-history logs URL for a failed concern.

    Args:
        concern: Doozer health concern containing failed-build metadata.
        encode_after: Whether to URL-encode the timestamp query value.
        strict: Raise on missing build metadata when true; otherwise return empty.

    Returns:
        ART build-history logs URL, or an empty string when metadata is incomplete.
    """
    nvr = concern.get("latest_failed_nvr")
    record_id = concern.get("latest_failed_build_record_id")
    failed_time = concern.get("latest_failed_build_time")
    if not nvr or not record_id or not failed_time:
        if strict:
            missing = next(
                key
                for key, value in (
                    ("latest_failed_nvr", nvr),
                    ("latest_failed_build_record_id", record_id),
                    ("latest_failed_build_time", failed_time),
                )
                if not value
            )
            raise KeyError(missing)
        return ""

    timestamp = datetime.fromisoformat(str(failed_time)).astimezone(timezone.utc)
    formatted = timestamp.strftime("%a, %d %b %Y %H:%M:%S GMT")
    if encode_after:
        formatted = quote(formatted)
    return f"{ART_BUILD_HISTORY_URL}/logs?nvr={nvr}&record_id={record_id}&after={formatted}"


def slack_url_text(url: str, text: str) -> str:
    """
    Format a URL as Slack link text.

    Args:
        url: URL to link.
        text: Display text.

    Returns:
        Slack link markup.
    """
    return f"<{quote(url, safe=':/?&=+%.-')}|{text}>"


def format_build_concern_line(
    concern: dict,
    search_url: str | Callable[[], str],
    logs_url: str | Callable[[], str],
    url_text: Callable[[str, str], str] = slack_url_text,
    jira_link: str = "",
) -> str:
    """
    Format one build-health concern for a Slack report.

    Args:
        concern: Doozer health concern.
        search_url: ART build-history URL or a callback that builds it.
        logs_url: ART build logs URL or a callback that builds it.
        url_text: Slack link formatter.
        jira_link: Optional Jira-link suffix for the concern.

    Returns:
        Slack-formatted build concern line.
    """
    image_name = concern["image_name"]
    code = concern.get("code")
    if code == ConcernCode.NEVER_BUILT.value:
        return f"- `{image_name}`: No builds attempted during last {DELTA_DAYS} days"

    history_url = search_url() if callable(search_url) else search_url
    message = f"- `{image_name}`: {url_text(history_url, 'Build history')}"
    if code in (ConcernCode.LATEST_ATTEMPT_FAILED.value, ConcernCode.FAILING_AT_LEAST_FOR.value):
        failure_logs_url = logs_url() if callable(logs_url) else logs_url
        if failure_logs_url:
            message += f" | {url_text(failure_logs_url, 'Latest failure logs')}"
        message += jira_link

    if code == ConcernCode.FAILING_AT_LEAST_FOR.value:
        return f"{message} - Failing for at least {LIMIT_BUILD_RESULTS} attempts"
    return f"{message} - Latest attempt failed ({concern.get('latest_success_idx', '?')} attempts since last success)"


def filter_failure_concerns(concerns: list[dict]) -> list[dict]:
    """
    Remove successful-build entries from Doozer health concerns.

    Args:
        concerns: Raw Doozer health concerns.

    Returns:
        Concerns that represent a failed or never-built image.
    """
    return [concern for concern in concerns if concern.get("code") != ConcernCode.LATEST_BUILD_SUCCEEDED.value]


def format_counter_failure_line(
    image_name: str,
    failure: dict,
    url_text: Callable[[str, str], str] = slack_url_text,
    url_label: str = "Last failure",
    primary_url_key: str | None = None,
) -> str:
    """
    Format one Redis counter failure entry.

    Args:
        image_name: Image distgit key.
        failure: Redis failure metadata.
        url_text: Slack link formatter.
        url_label: Display text for the last-failure link.
        primary_url_key: Optional Redis metadata key to use for the primary link.

    Returns:
        Slack-formatted failure line without a trailing newline.
    """
    failure_count = failure.get("failure_count", 0)
    suffix = "" if failure_count == 1 else "s"
    line = f"- `{image_name}`: Failed {failure_count} time{suffix}"
    url = failure.get(primary_url_key) if primary_url_key else failure.get("pipeline_url") or failure.get("jenkins_url")
    if url:
        line += f" ({url_text(url, url_label)})"
    return line


def format_counter_failure_section(
    title: str,
    failures: dict[str, dict],
    url_text: Callable[[str, str], str] = slack_url_text,
    url_label: str = "Last failure",
    include_pipeline_url: bool = False,
    primary_url_key: str | None = None,
) -> str:
    """
    Format a section containing Redis counter failures.

    Args:
        title: Section heading.
        failures: Image failures keyed by image name.
        url_text: Slack link formatter.
        url_label: Display text for the last-failure link.
        include_pipeline_url: Add a pipeline link after the standard failure link.
        primary_url_key: Optional Redis metadata key to use for the primary link.

    Returns:
        Slack-formatted section.
    """
    lines = [f"*{title} ({len(failures)}):*"]
    for image_name, failure in sorted(failures.items()):
        line = format_counter_failure_line(image_name, failure, url_text, url_label, primary_url_key)
        if include_pipeline_url and failure.get("pipeline_url"):
            line += f" | {url_text(failure['pipeline_url'], 'Pipeline')}"
        lines.append(line)
    return "\n".join(lines)


def format_rebase_failure_section(
    failures: dict[str, dict],
    url_text: Callable[[str, str], str] = slack_url_text,
    url_label: str = "Last failure job",
    include_build_system: bool = False,
    include_pipeline_url: bool = False,
) -> str:
    """
    Format a section containing Redis rebase failures.

    Args:
        failures: Image rebase failures keyed by image name.
        url_text: Slack link formatter.
        url_label: Display text for the last-failure link.
        include_build_system: Include each failure's build system in its line.
        include_pipeline_url: Use a pipeline URL when a Jenkins URL is absent.

    Returns:
        Slack-formatted rebase section.
    """
    lines = [f"*Rebase Failures ({len(failures)}):*"]
    for image_name, failure in sorted(failures.items()):
        failure_count = failure.get("failure_count", 0)
        suffix = "" if failure_count == 1 else "s"
        build_system = f" ({failure.get('build_system', 'unknown')})" if include_build_system else ""
        line = f"- `{image_name}`{build_system}: Failed {failure_count} time{suffix}"
        if include_pipeline_url:
            url = failure.get("pipeline_url") or failure.get("jenkins_url")
        else:
            url = failure.get("jenkins_url")
        if url:
            line += f" ({url_text(url, url_label)})"
        lines.append(line)
    return "\n".join(lines)


def get_multi_failure_images(report: list[dict], group_prefix: str = "openshift-") -> dict[str, list[dict]]:
    """
    Group images with multiple consecutive build failures by version.

    Args:
        report: Flat list of Doozer health concerns.
        group_prefix: Group prefix used to extract the version.

    Returns:
        Version-keyed build concerns with more than one consecutive failure.
    """
    result: dict[str, list[dict]] = {}
    failure_codes = {ConcernCode.LATEST_ATTEMPT_FAILED.value, ConcernCode.FAILING_AT_LEAST_FOR.value}
    for concern in report:
        if concern.get("code") not in failure_codes or concern.get("latest_success_idx", 0) <= 1:
            continue
        group = concern.get("group", "")
        version = group.removeprefix(group_prefix)
        result.setdefault(version, []).append(concern)
    return result


def get_multi_rebase_failures(rebase_failures: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """
    Keep only versions containing images with multiple consecutive rebase failures.

    Args:
        rebase_failures: Version-keyed rebase failures.

    Returns:
        Version-keyed failures with failure counts greater than one.
    """
    result = {}
    for version, failures in rebase_failures.items():
        multi_failures = {image: info for image, info in failures.items() if info.get("failure_count", 0) > 1}
        if multi_failures:
            result[version] = multi_failures
    return result
