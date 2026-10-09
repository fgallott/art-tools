import asyncio
from typing import Optional

import click
from artcommonlib import exectools
from artcommonlib.constants import ACTIVE_OCP_VERSIONS
from doozerlib.cli.images_health import LIMIT_BUILD_RESULTS, ConcernCode
from doozerlib.constants import ART_BUILD_FAILURES_URL

from pyartcd import util
from pyartcd.cli import cli, click_coroutine, pass_runtime
from pyartcd.constants import OCP_BUILD_DATA_URL
from pyartcd.pipelines.image_health_common import (
    build_group_param,
    build_history_url,
    build_logs_url,
    filter_failure_concerns,
    filter_image_list,
    filter_image_names,
    format_build_concern_line,
    format_rebase_failure_section,
    get_counter_failures,
    get_filtered_rebase_failures,
    get_multi_failure_images,
    get_multi_rebase_failures,
    get_valid_images_from_doozer,
    run_images_health,
    slack_url_text,
)
from pyartcd.runtime import Runtime

OKD_GROUP_TEMPLATE = "okd-{}"


class ImagesHealthPipeline:
    def __init__(
        self,
        runtime: Runtime,
        versions: str,
        send_to_release_channel: bool,
        send_to_okd_channel: bool,
        ping_chai_bot: bool,
        data_path: str,
        data_gitref: str,
        image_list: str,
        assembly: str,
    ):
        self.runtime = runtime
        self._versions_param = versions
        self.doozer_working = self.runtime.working_dir / "doozer_working"
        self.send_to_release_channel = send_to_release_channel
        self.send_to_okd_channel = send_to_okd_channel
        self.ping_chai_bot = ping_chai_bot
        self.data_path = data_path
        self.data_gitref = data_gitref
        self.image_list = image_list.split(',') if image_list else []
        self.assembly = assembly if assembly else 'stream'

        self.report = []
        self.slack_client = self.runtime.new_slack_client()
        self.scanned_versions = []
        self.rebase_failures = {}  # version -> {image: {failure_count, url}}

    def _doozer_base_command(self, version: str) -> list[str]:
        group_param = f'openshift-{version}'
        if self.data_gitref:
            group_param += f'@{self.data_gitref}'
        return [
            'doozer',
            f'--working-dir={self.doozer_working}-{version}',
            f'--data-path={self.data_path}',
            f'--group={group_param}',
            f'--assembly={self.assembly}',
            '--build-system=konflux',
            '--variant=okd',
        ]

    async def _resolve_versions(self) -> list[str]:
        if self._versions_param:
            candidates = [v.strip() for v in self._versions_param.split(',') if v.strip()]
        else:
            candidates = list(ACTIVE_OCP_VERSIONS)
            self.runtime.logger.info(
                'No --versions provided; probing ACTIVE_OCP_VERSIONS for okd.enabled in build-data'
            )

        enabled_versions = []
        for version in candidates:
            if await util.is_okd_version_enabled(self._doozer_base_command(version)):
                enabled_versions.append(version)
            else:
                self.runtime.logger.info(
                    'Version %s is not enabled for OKD (set okd.enabled: true in group.yml on openshift-%s). Skipping.',
                    version,
                    version,
                )
        return enabled_versions

    async def run(self):
        self.versions = await self._resolve_versions()
        if not self.versions:
            self.runtime.logger.info('No OKD-enabled versions to monitor; skipping health report')
            return
        await asyncio.gather(*(self.get_report(v) for v in self.versions))
        await asyncio.gather(*(self.get_rebase_failures(v) for v in self.versions))
        self.runtime.logger.info('Found %s concerns', len(self.report))

        if self.send_to_release_channel:
            for version in self.scanned_versions:
                await self.notify_release_channel(version)

        if self.send_to_okd_channel:
            await self.notify_okd_channel()

        if self.ping_chai_bot:
            multi_build_failures = self._get_multi_failure_images()
            multi_rebase_failures = self._get_multi_rebase_failures()
            all_versions = set(multi_build_failures) | set(multi_rebase_failures)
            for version in all_versions:
                await self.notify_chai_bot(
                    version,
                    multi_build_failures.get(version, []),
                    multi_rebase_failures.get(version, {}),
                )

    async def get_report(self, version: str) -> Optional[list]:
        group = OKD_GROUP_TEMPLATE.format(version)
        failures = await get_counter_failures('build-failure', group, self.runtime.logger)

        failing_images = filter_image_list(set(failures), set(self.image_list))

        if not failing_images:
            self.runtime.logger.info('No build failures in Redis for %s; skipping BigQuery scan', group)
            self.scanned_versions.append(version)
            return

        # Filter failing_images to only include images that exist in current ocp-build-data
        doozer_working = f'{self.doozer_working}-{version}'
        valid_images = await self._get_valid_images(version, doozer_working)
        filtered_failing_images = filter_image_names(failing_images, valid_images, group, self.runtime.logger)

        if not filtered_failing_images:
            self.runtime.logger.info(
                'No valid failing images remain for %s after filtering; skipping BigQuery scan', group
            )
            self.scanned_versions.append(version)
            return

        self.runtime.logger.info(
            'Redis reports %d failing image(s) for %s; querying BigQuery for details',
            len(filtered_failing_images),
            group,
        )

        cmd = [
            'doozer',
            f'--working-dir={doozer_working}',
            f'--data-path={self.data_path}',
            '--variant=okd',
            build_group_param(f'openshift-{version}', self.data_gitref),
        ]

        after_command = [f'--group={group}']
        if self.assembly:
            after_command.append(f'--assembly={self.assembly}')

        report, out = await run_images_health(
            cmd,
            filtered_failing_images,
            tuple(after_command),
            command_runner=exectools.cmd_gather_async,
        )

        self.runtime.logger.info('images:health output for %s:\n%s', group, out)
        self.report.extend(report)
        self.scanned_versions.append(version)

    async def get_rebase_failures(self, version: str):
        """
        Fetch OKD rebase failure data from Redis for a specific version.
        Filters out images that don't exist in current ocp-build-data.
        Populates self.rebase_failures[version] with {image: {failure_count, url}}.

        Arg(s):
            version (str): OKD version (e.g., "4.21")
        """
        group = f'okd-{version}'
        doozer_working = f'{self.doozer_working}-{version}'
        valid_images = await self._get_valid_images(version, doozer_working)
        filtered_failures = await get_filtered_rebase_failures(
            group, valid_images, self.runtime.logger, ['konflux']
        )
        self.rebase_failures[version] = filtered_failures

    async def _get_valid_images(self, version: str, doozer_working: str) -> set[str]:
        """
        Get the set of valid image names for a given OKD version from ocp-build-data.

        Arg(s):
            version (str): OKD version (e.g., "4.21")
            doozer_working (str): Doozer working directory path
        Return Value(s):
            set[str]: Set of valid image distgit keys
        """
        return await get_valid_images_from_doozer(
            self.runtime,
            group=f'openshift-{version}',
            working_dir=doozer_working,
            data_path=self.data_path,
            data_gitref=self.data_gitref,
            variant='okd',
        )

    async def notify_release_channel(self, version):
        """
        Send notifications to version-specific release channel (e.g., #art-release-4-21) for a specific OKD version.
        Uses the same channels as OCP releases.
        Filters out LATEST_BUILD_SUCCEEDED concerns (successful builds not reported).
        Posts parent message with concern count, details in thread.
        Includes rebase failure information from Redis.

        Arg(s):
            version (str): OKD version (e.g., "4.21")
        """
        # Bind to version-specific channel (same as OCP: #art-release-4-21)
        channel = f"#art-release-{version.replace('.', '-')}"
        self.slack_client.bind_channel(channel)

        # Filter concerns for this version, excluding successful builds
        concerns = [
            concern for concern in filter_failure_concerns(self.report) if concern.get('group') == f'openshift-{version}'
        ]

        # Get rebase failures for this version
        rebase_failures = self.rebase_failures.get(version, {})

        version_tag = f'`okd-{version}`'
        if self.assembly != 'stream':
            version_tag += f' (assembly `{self.assembly}`)'

        # If no concerns and no rebase failures, report all healthy
        if not concerns and not rebase_failures:
            await self.slack_client.say(f':white_check_mark: All OKD images are healthy for {version_tag}')
            return

        # Build summary message
        summary_parts = []
        if concerns:
            summary_parts.append(self.get_component_tag(concerns))
        if rebase_failures:
            rebase_count = len(rebase_failures)
            summary_parts.append(f'{rebase_count} image{"s" if rebase_count > 1 else ""} with rebase failures')

        # Post parent message with concern count
        issues = '\n- '.join(summary_parts)
        response = await self.slack_client.say(
            f':alert: There are some issues to look into for OKD {version_tag}:\n- {issues}'
        )

        # Post detailed report in thread
        report = ''

        # Add build concerns
        if concerns:
            report += '*Build Issues:*\n'
            for concern in concerns:
                report += f'{self.get_message_for_release(concern)}\n'
            report += '\n'

        # Add rebase failures
        if rebase_failures:
            report += format_rebase_failure_section(rebase_failures, self.url_text)

        await self.slack_client.say(report, thread_ts=response['ts'])

    async def notify_okd_channel(self):
        """
        Send a summary message to #art-okd-release channel with a link to the dashboard.
        Instead of posting detailed failure lists (which can overflow Jira API limits),
        we now only post a summary grouped by version and direct users to the art-build-failures dashboard.
        """
        self.slack_client.bind_channel('#art-okd-release')

        # Group build concerns by OKD group
        version_build_failures = {}
        for concern in self.report:
            if concern['code'] in (ConcernCode.NEVER_BUILT.value, ConcernCode.LATEST_BUILD_SUCCEEDED.value):
                # We don't report NEVER_BUILT concerns to art-okd-release. Latest built succeeded is not a concern.
                continue
            group = concern['group']
            # Transform openshift-X.Y to okd-X.Y
            okd_group = group.replace('openshift-', 'okd-')
            version_build_failures.setdefault(okd_group, []).append(concern)

        # Rebase failures are already keyed by version in self.rebase_failures
        # Need to collect all groups that have any kind of failure

        # If no concerns and no rebase failures, report all healthy
        if not version_build_failures and not self.rebase_failures:
            await self.slack_client.say(':white_check_mark: All OKD images are healthy for all monitored releases')
            return

        # Collect all groups that have any kind of failure
        all_groups = set()
        all_groups.update(version_build_failures.keys())
        all_groups.update(f'okd-{v}' for v in self.rebase_failures.keys())

        # Build message grouped by version
        message_parts = [':alert: There are some issues to look into for OKD builds:\n']

        for okd_group in sorted(all_groups):
            version = okd_group.replace('okd-', '')
            group_summary = []

            # Build failures for this group
            build_fails = version_build_failures.get(okd_group, [])
            if build_fails:
                n = len(build_fails)
                group_summary.append(f'{n} image{"s" if n > 1 else ""} with build failures')

            # Rebase failures for this group
            rebase_fails = self.rebase_failures.get(version, {})
            if rebase_fails:
                n = len(rebase_fails)
                group_summary.append(f'{n} image{"s" if n > 1 else ""} with rebase failures')

            if group_summary:
                message_parts.append(f'\n*{okd_group}*:')
                for item in group_summary:
                    message_parts.append(f'- {item}')

        # Link to art-build-failures dashboard
        dashboard_url = ART_BUILD_FAILURES_URL
        message_parts.append(
            f'\nFor detailed information, please check the {self.url_text(dashboard_url, "ART Build Failures Dashboard")}'
        )

        message = '\n'.join(message_parts)
        await self.slack_client.say(message, link_build_url=False, unfurl_links=False, unfurl_media=False)

    def get_message_for_release(self, concern: dict):
        """
        Format a concern message for the release channel (#art-okd-release).

        Arg(s):
            concern (dict): Concern data from doozer images:health
        Return Value(s):
            str: Formatted message with links and details
        """
        return format_build_concern_line(
            concern,
            lambda: self.get_search_url(concern),
            lambda: self.get_logs_url(concern),
            self.url_text,
        )

    def get_message_for_okd_channel(self, concern: dict):
        """
        Format a concern message for the general #art-okd-release channel.
        Shows group (version) and failure details with links.

        Arg(s):
            concern (dict): Concern data from doozer images:health
        Return Value(s):
            str: Formatted message with links and details
        """
        code = concern['code']
        group = concern['group']
        # Transform openshift-X.Y to okd-X.Y for display
        okd_group = group.replace('openshift-', 'okd-')

        art_dash_link = build_history_url(concern['image_name'], okd_group, 'stream')
        logs_link = self.url_text(self.get_logs_url(concern), "logs")

        message = f'{self.url_text(art_dash_link, f"{okd_group}")}: '

        if code == ConcernCode.FAILING_AT_LEAST_FOR.value:
            message += f'more than {LIMIT_BUILD_RESULTS} failures ({logs_link}).'
        else:  # ConcernCode.LATEST_ATTEMPT_FAILED
            message += f'{concern["latest_success_idx"]} failures ({logs_link}).'

        return message

    @staticmethod
    def get_search_url(concern):
        """
        Build the OKD build history search page URL for a component.

        Arg(s):
            concern (dict): Concern data containing image name and group
        Return Value(s):
            str: URL to the build history search page
        """
        image_name = concern['image_name']
        group = concern['group']
        # Transform openshift-X.Y to okd-X.Y for the OKD dashboard
        okd_group = group.replace('openshift-', 'okd-')
        return build_history_url(image_name, okd_group, 'stream')

    @staticmethod
    def get_logs_url(concern):
        """
        Build the OKD build history logs URL for a failed build.

        Arg(s):
            concern (dict): Concern data containing build failure details
        Return Value(s):
            str: URL to the build logs
        """
        return build_logs_url(concern, encode_after=False)

    @staticmethod
    def get_component_tag(report):
        """
        Create a component count tag for the summary message.

        Arg(s):
            report (list or dict): Either a list of concerns or dict of concerns by image
        Return Value(s):
            str: Formatted component count message
        """
        n_components = len(report)

        if n_components > 1:
            return f'{n_components} components have failed'
        else:
            return '1 component has failed'

    def url_text(self, url, text):
        """
        Slack requires URLs to be encoded in a specific way when using the <url|text> format.
        This function ensures that the URL is properly encoded while keeping certain characters safe.

        Arg(s):
            url (str): The URL to encode
            text (str): The display text for the link
        Return Value(s):
            str: Slack-formatted link <url|text>
        """
        try:
            return slack_url_text(url, text)

        except Exception as e:
            self.runtime.logger.warning('invalid URL: %s', e)

    def _get_multi_failure_images(self) -> dict[str, list[dict]]:
        """
        Filter self.report for images with >1 consecutive failure in a single group.
        Only includes LATEST_ATTEMPT_FAILED and FAILING_AT_LEAST_FOR concern codes
        where latest_success_idx > 1.

        Return Value(s):
            dict[str, list[dict]]: Version -> list of failing concerns
        """
        return get_multi_failure_images(self.report)

    def _get_multi_rebase_failures(self) -> dict[str, dict[str, dict]]:
        """
        Filter self.rebase_failures for images with >1 consecutive rebase failure.

        Return Value(s):
            dict[str, dict[str, dict]]: Version -> {image_name: failure_info}
        """
        return get_multi_rebase_failures(self.rebase_failures)

    def _build_chai_bot_prompt(
        self, version: str, failing_concerns: list[dict], rebase_failures: dict[str, dict] | None = None
    ) -> str:
        """
        Build structured prompt for @chai-bot to fix OKD build and rebase failures.

        Arg(s):
            version (str): OKD version (e.g., "4.21")
            failing_concerns (list[dict]): List of concern dicts with build failure info
            rebase_failures (dict[str, dict]): Map of image_name -> failure_info for rebase failures
        Return Value(s):
            str: Formatted prompt for chai-bot
        """
        okd_group = f'okd-{version}'
        group = f'openshift-{version}'
        prompt_parts = [f'Please investigate and fix OKD build failures for *{okd_group}*.\n']

        if failing_concerns:
            prompt_parts.append('*Failing Images (build failures):*')
            for concern in failing_concerns:
                image_name = concern['image_name']
                failure_count = concern.get('latest_success_idx', 0)
                logs_url = self.get_logs_url(concern)
                search_url = self.get_search_url(concern)

                prompt_parts.append(
                    f'- `{image_name}`: {failure_count} consecutive failures '
                    f'({self.url_text(search_url, "history")} | {self.url_text(logs_url, "latest logs")})'
                )

        if rebase_failures:
            prompt_parts.append('\n*Failing Images (rebase failures):*')
            for image_name, failure_info in sorted(rebase_failures.items()):
                failure_count = failure_info.get('failure_count', 0)
                jenkins_url = failure_info.get('jenkins_url', '')
                line = f'- `{image_name}`: {failure_count} consecutive rebase failures'
                if jenkins_url:
                    line += f' ({self.url_text(jenkins_url, "last failure job")})'
                prompt_parts.append(line)

        prompt_parts.extend(
            [
                '\n*Instructions:*',
                '0. *Pre-check:* Before each image query Jira: `project = ART AND summary ~ "Image build failure: <image_name>"'
                f' AND summary ~ "({group})" AND statusCategory != Done`.'
                ' If ticket has `art:bot-skip-auto-fix` label, skip that image and report with ticket link.',
                '1. Analyze failure logs. Fix via ocp-build-data metadata changes only (Dockerfile patches, env vars, build args).'
                ' *DO NOT* push to upstream repos. Only proceed if confidence >= 90%.',
                f'2. If proceeding: push fix to ocp-build-data fork for `openshift-{version}` (`--variant=okd`).'
                ' *Changes MUST be under `okd:` stanza only.*'
                ' *Commit message MUST start with `scan-sources-konflux:noop`.*'
                f' Then trigger Jenkins build for `{okd_group}` *without user confirmation* (`ASSEMBLY=stream`, `IGNORE_LOCKS=true`).',
                '   - *Max 3 fix attempts per image.* After 3 failures: stop, report all tried approaches,'
                ' apply `art:bot-skip-auto-fix` to the Jira ticket (ART can remove to retry).',
                '   - On success: report fixed images, changes made, and test build links; *ask user to approve PR*.',
                '   - On approval: file PR attributed to bot (not Slack profile), *squash merge* (merge commits nullify `:noop`).'
                ' PR description *must include* full clickable Jenkins build URL'
                ' (e.g. `https://art-jenkins.apps.prod-stable-spoke1-dc-iad2.itup.redhat.com/job/aos-cd-builds/job/build%252Fokd/BUILD_NUMBER/`)'
                ' and full clickable Slack thread `https://` URL.',
                '3. If confidence < 90% or upstream change required: diagnostic report only,'
                ' suggest `art:bot-skip-auto-fix` label on Jira ticket, defer to humans.',
                '\n*Target only OKD builds. All fixes must be ocp-build-data metadata changes only.*',
            ]
        )

        return '\n'.join(prompt_parts)

    async def notify_chai_bot(
        self, version: str, failing_concerns: list[dict], rebase_failures: dict[str, dict] | None = None
    ):
        """
        Post prompt to chai-bot channel requesting automated fix attempts.

        Arg(s):
            version (str): OKD version (e.g., "4.21")
            failing_concerns (list[dict]): List of build failure concerns with >1 consecutive failure
            rebase_failures (dict[str, dict]): Map of image_name -> failure_info for rebase failures with >1 failure
        """
        if not failing_concerns and not rebase_failures:
            return

        prompt = self._build_chai_bot_prompt(version, failing_concerns, rebase_failures or {})
        self.slack_client.bind_channel('#team-art-chai-bot')

        message = f'<@U0AKNPBBVT7> {prompt}'
        await self.slack_client.say(message, unfurl_links=False, unfurl_media=False)
        self.runtime.logger.info(
            'Notified chai-bot in #team-art-chai-bot for %d build failure(s) and %d rebase failure(s) in okd-%s',
            len(failing_concerns),
            len(rebase_failures) if rebase_failures else 0,
            version,
        )


@cli.command('okd-images-health')
@click.option('--versions', required=False, default='', help='OCP versions to scan')
@click.option('--send-to-release-channel', is_flag=True, help='If true, send output to #art-release-4-<version>')
@click.option('--send-to-okd-channel', is_flag=True, help='If true, send aggregated notification to #art-okd-release')
@click.option(
    '--ping-chai-bot', is_flag=True, help='If true, notify @chai-bot in #team-art-chai-bot for multi-failure images'
)
@click.option(
    '--data-path',
    required=False,
    default=OCP_BUILD_DATA_URL,
    help='ocp-build-data fork to use (e.g. assembly definition in your own fork)',
)
@click.option('--data-gitref', required=False, default='', help='Doozer data path git [branch / tag / sha] to use')
@click.option(
    '--image-list',
    required=False,
    help='Comma/space-separated list to include/exclude per --image-build-strategy (e.g. ironic,hypershift)',
)
@click.option('--assembly', required=False, default='', help='Assembly override')
@pass_runtime
@click_coroutine
async def okd_images_health(
    runtime: Runtime,
    versions: str,
    send_to_release_channel: bool,
    send_to_okd_channel: bool,
    ping_chai_bot: bool,
    data_path: str,
    data_gitref: str,
    image_list: str,
    assembly: str,
):
    await ImagesHealthPipeline(
        runtime,
        versions,
        send_to_release_channel,
        send_to_okd_channel,
        ping_chai_bot,
        data_path,
        data_gitref,
        image_list,
        assembly,
    ).run()
