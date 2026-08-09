from __future__ import annotations

import re
from collections.abc import Iterable
from ipaddress import ip_address
from urllib.parse import ParseResult, unquote, urlparse

from tld import get_tld

NONSPACE_TOKEN_PATTERN = re.compile(r"\S+")
WRAPPED_SPAN_PATTERN = re.compile(
    r"```[^\n`]*\n(?P<fenced>.*?)```"
    r"|\[[^\]\n]*\]\((?P<markdown_destination>(?:\\.|[^()\n]|\((?:\\.|[^()\n])*\))*)\)"
    r"|`(?P<backtick>[^`]*)`"
    r"|\"(?P<double>[^\"]*)\""
    r"|'(?P<single>[^']*)'"
    r"|“(?P<curly_double>.*?)”"
    r"|‘(?P<curly_single>.*?)’"
    r"|「(?P<cjk_single>.*?)」"
    r"|『(?P<cjk_double>.*?)』"
    r"|【(?P<cjk_square>.*?)】"
    r"|（(?P<fullwidth_parentheses>.*?)）"
    r"|《(?P<cjk_angle>.*?)》"
    r"|«(?P<guillemet>.*?)»"
    r"|‹(?P<single_guillemet>.*?)›"
    r"|„(?P<german_quote>.*?)“"
    r"|｢(?P<halfwidth_corner>.*?)｣"
    r"|〈(?P<cjk_chevron>.*?)〉"
    r"|〔(?P<cjk_tortoise>.*?)〕"
    r"|［(?P<fullwidth_square>.*?)］"
    r"|\*\*(?P<bold>.*?)\*\*"
    r"|__(?P<strong>.*?)__"
    r"|(?<!_)_(?!_)(?P<emphasis>.*?)(?<!_)_(?!_)"
    r"|~~(?P<strike>.*?)~~"
    r"|<(?P<angle>[^<>]*)>",
    re.DOTALL,
)
EXPLICIT_SCHEME_PATTERN = re.compile(
    r"^(?:(?:[a-z][a-z0-9+.-]*):/{2}|(?:file|data):)", re.IGNORECASE
)
OPAQUE_URI_PATTERN = re.compile(
    r"^(?:bitcoin|did|ethereum|geo|lightning|magnet|mailto|monero|news|otpauth|"
    r"sip|sips|sms|tel|urn|webcal|xmpp):\S+$",
    re.IGNORECASE,
)
SCP_REMOTE_PATTERN = re.compile(
    r"^(?P<user>[^@\s:]+)@(?P<host>[^@\s:/]+):(?P<path>\S+)$", re.IGNORECASE
)
SCP_HOST_REMOTE_PATTERN = re.compile(r"^(?P<host>[^@\s:/]+):(?P<path>[^\s:]+)$", re.IGNORECASE)
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
SCOPED_PACKAGE_PATTERN = re.compile(r"^@[a-z0-9_.-]+/[a-z0-9_.-]+$", re.IGNORECASE)
DOTTED_RUNTIME_PATH_PATTERN = re.compile(
    r"^(?:v?\d+(?:\.\d+){1,4}|[a-z][a-z0-9_-]*\d*(?:\.\d+)+)/[a-z0-9_.-]+$",
    re.IGNORECASE,
)
REVERSE_DOMAIN_NAMESPACE_PATTERN = re.compile(
    r"^(?:com|org|net|io|edu)\.[a-z0-9_-]+(?:\.[a-z0-9_-]+)*/[a-z0-9_./-]+$",
    re.IGNORECASE,
)
LOCAL_PATH_START_PATTERN = (
    r"(?:[a-z]:[\\/]"
    r"|\\\\"
    r"|/(?!/)(?=\S)"
    r"|~(?:[a-z0-9._-]+)?[\\/]"
    r"|\.{1,2}[\\/]"
    r"|\$(?:HOME|PWD)[\\/]"
    r"|\$\{(?:HOME|PWD)\}[\\/]"
    r"|%(?:USERPROFILE|APPDATA|LOCALAPPDATA|HOMEPATH)%[\\/])"
)
LOCAL_PATH_PATTERN = re.compile(rf"^{LOCAL_PATH_START_PATTERN}", re.IGNORECASE)
PATH_LABEL_WORD_PATTERN = r"(?:path|file|cwd|dir|directory|location|路径|路徑|文件|目录|目錄|位置)"
URL_LABEL_WORD_PATTERN = r"(?:url|link|source|host|网址|鏈接|链接|來源|来源|主机|主機)"
URL_LABEL_PATTERN = re.compile(
    rf"^{URL_LABEL_WORD_PATTERN}[\"']?\s*[:=：＝]\s*[\"']?",
    re.IGNORECASE,
)
LEADING_LABEL_PATTERN = re.compile(
    r"^(?:url|link|source|host|path|file|cwd|dir|directory|location|"
    r"网址|鏈接|链接|來源|来源|主机|主機|路径|路徑|文件|目录|目錄|位置)"
    r"[\"']?\s*[:=：＝]\s*[\"']?",
    re.IGNORECASE,
)
EMBEDDED_LOCAL_PATH_PATTERN = re.compile(
    rf"(?:^|[^\w/\\]){LOCAL_PATH_START_PATTERN}",
    re.IGNORECASE,
)
ROOTED_LOCAL_PATH_START_PATTERN = re.compile(
    rf"(?P<prefix>^|[^\w/\\])(?P<path>{LOCAL_PATH_START_PATTERN})",
    re.IGNORECASE | re.MULTILINE,
)
LABELED_LOCAL_PATH_START_PATTERN = re.compile(
    rf"(?P<prefix>^|[^\w/\\])"
    rf"{PATH_LABEL_WORD_PATTERN}[\"'”’」』]*\s*[:=：＝]\s*[\"'“‘「『`*_~<]*"
    r"(?P<path>[^\s,;]+)",
    re.IGNORECASE | re.MULTILINE,
)
LABELED_URL_START_PATTERN = re.compile(
    rf"(?P<prefix>^|[^\w/\\]){URL_LABEL_WORD_PATTERN}"
    r"[\"'”’」』]*\s*[:=：＝]\s*[\"'“‘「『`*_~<]*"
    r"(?P<url>[^\s]+)",
    re.IGNORECASE | re.MULTILINE,
)
DETECTED_URL_START_PATTERN = re.compile(
    r"(?P<prefix>^|[^\w/\\])(?P<url>[^\s]*[\\/][^\s]+)",
    re.IGNORECASE | re.MULTILINE,
)
PATH_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"(?:[,;，；]\s*(?:(?:and|or|with|then|versus|vs\.?|和|与|與|及|然后|然後|"
    r"と|そして|または|그리고|및|또는)\s+)?|"
    r"\s+(?:and|or|with|then|versus|vs\.?|和|与|與|及|然后|然後|"
    r"と|そして|または|그리고|및|또는)\s+)",
    re.IGNORECASE,
)
FRESH_EXPLICIT_URL_PATTERN = re.compile(
    r"[,;，；](?=(?://|[a-z][a-z0-9+.-]*://))",
    re.IGNORECASE,
)
ENGLISH_QUERY_START_PATTERN = re.compile(
    r"^(?:analy[sz]e|calculate|check|compare|explain|find|get|inspect|learn|open|"
    r"read|research|review|search|show|summari[sz]e|verify)\b",
    re.IGNORECASE,
)
MULTILINGUAL_QUERY_TERM_PATTERN = re.compile(
    r"(?:比较|比較|研究|搜索|搜尋|查找|解释|解釋|分析|总结|總結|验证|驗證|"
    r"比較|検索|説明|検証|要約|비교|검색|연구|설명|분석|검증)"
)
WRAPPING_PUNCTUATION = "\"'`()[]{}<>,;.!?*“”‘’「」『』【】（）《》«»‹›„｢｣〈〉〔〕［］_~"
TECHNICAL_FILE_SUFFIXES = {
    "c",
    "cc",
    "cpp",
    "css",
    "go",
    "h",
    "hpp",
    "html",
    "java",
    "js",
    "json",
    "jsx",
    "md",
    "py",
    "rs",
    "sh",
    "toml",
    "ts",
    "tsx",
    "txt",
    "xml",
    "yaml",
    "yml",
}


def _privacy_decode(value: str) -> str:
    """Decode nested percent escapes with a small, deterministic work bound."""

    decoded = value
    for _ in range(3):
        updated = unquote(decoded)
        if updated == decoded:
            break
        decoded = updated
    return decoded


def _looks_like_query_clause(value: str) -> bool:
    stripped = value.strip()
    return bool(
        ENGLISH_QUERY_START_PATTERN.match(stripped)
        or MULTILINGUAL_QUERY_TERM_PATTERN.search(stripped)
    )


def _has_relative_path_evidence(value: str) -> bool:
    decoded = _privacy_decode(value).strip()
    if LOCAL_PATH_PATTERN.match(decoded) or "\\" in decoded:
        return True
    if decoded.count("/") >= 2:
        return True
    return bool(
        re.search(
            r"(?:^|[/\s])[^/\s]+\.[a-z0-9]{1,12}(?:[\s\]})>”’」』】）》]*$)",
            decoded,
            re.IGNORECASE,
        )
    )


def _cleaned_token(token: str) -> str:
    cleaned = token.strip().strip(WRAPPING_PUNCTUATION)
    if "](" in cleaned:
        cleaned = cleaned.rsplit("](", 1)[-1]
    if "://" in cleaned:
        delimiter = cleaned.find("://")
        start = delimiter
        while start > 0 and (cleaned[start - 1].isalnum() or cleaned[start - 1] in "+.-"):
            start -= 1
        cleaned = cleaned[start:]
    explicit_scheme = bool(EXPLICIT_SCHEME_PATTERN.match(cleaned))
    if not explicit_scheme:
        cleaned = LEADING_LABEL_PATTERN.sub("", cleaned)
    for separator in () if explicit_scheme else ("=", "＝", ":", "："):
        if separator not in cleaned:
            continue
        _label, suffix = cleaned.split(separator, 1)
        suffix = suffix.strip(WRAPPING_PUNCTUATION)
        authority = suffix.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        if (
            suffix.startswith("//")
            or EXPLICIT_SCHEME_PATTERN.match(suffix)
            or (
                any(marker in suffix for marker in ("/", "?", "#"))
                and ("." in authority or authority.casefold() == "localhost")
            )
        ):
            cleaned = suffix
            break
    return cleaned.strip(WRAPPING_PUNCTUATION)


def _is_local_path_token(token: str) -> bool:
    cleaned = _cleaned_token(token)
    decoded = _privacy_decode(cleaned)
    file_uri = bool(re.match(r"^file:(?:[/\\]|%2f|%5c)", cleaned, re.IGNORECASE)) or bool(
        re.match(r"^file:[/\\]", decoded, re.IGNORECASE)
    )
    return bool(
        file_uri
        or LOCAL_PATH_PATTERN.match(cleaned)
        or LOCAL_PATH_PATTERN.match(decoded)
        or EMBEDDED_LOCAL_PATH_PATTERN.search(token)
        or EMBEDDED_LOCAL_PATH_PATTERN.search(decoded)
    )


def _wrapped_value(match: re.Match[str]) -> str:
    return next(value for value in match.groupdict().values() if value is not None)


def _is_ip_host(hostname: str) -> bool:
    try:
        ip_address(hostname)
    except ValueError:
        return False
    return True


def _is_private_network_host(hostname: str) -> bool:
    normalized = hostname.casefold().rstrip(".")
    try:
        address = ip_address(normalized)
    except ValueError:
        address = None
    if address is not None:
        return not address.is_global
    return normalized in {
        "gateway",
        "intranet",
        "internal",
        "localhost",
        "nas",
        "router",
    } or normalized.endswith((".home", ".internal", ".lan", ".local", ".localhost"))


def _has_public_suffix(hostname: str) -> bool:
    if not hostname or "." not in hostname or hostname.casefold() == "localhost":
        return False
    try:
        return bool(get_tld(f"http://{hostname}", fail_silently=True, fix_protocol=True))
    except (TypeError, ValueError):
        return False


def _path_clause_end(value: str, start: int, *, minimum_end: int | None = None) -> int:
    """Find a privacy-safe end for an unquoted path while retaining later prose.

    A comma or conjunction is a real boundary unless the next clause visibly
    resumes the same path (for example ``Research and Development/file``).
    This intentionally favors redaction for an ambiguous final path fragment.
    """

    line_end = value.find("\n", start)
    if line_end < 0:
        line_end = len(value)
    segment = value[start:line_end]
    safe_suffix = re.search(
        r"\s+(?:(?:current|docs|documentation|evidence|latest|now|official|"
        r"release|releases|today|version|versions)\s*){1,3}$",
        segment,
        re.IGNORECASE,
    )
    first_whitespace = re.search(r"\s+", segment)
    if (
        safe_suffix is not None
        and first_whitespace is not None
        and safe_suffix.start() == first_whitespace.start()
    ):
        return start + safe_suffix.start()
    boundaries = list(PATH_CLAUSE_BOUNDARY_PATTERN.finditer(segment))
    for index, boundary in enumerate(boundaries):
        if minimum_end is not None and start + boundary.start() < minimum_end:
            continue
        candidate_start = boundary.end()
        candidate_end = (
            boundaries[index + 1].start() if index + 1 < len(boundaries) else len(segment)
        )
        candidate = segment[candidate_start:candidate_end].strip()
        if not candidate:
            return start + boundary.start()
        first_token = candidate.split(None, 1)[0].strip(WRAPPING_PUNCTUATION)
        first_decoded = _privacy_decode(first_token)
        first_authority = first_decoded.split("/", 1)[0]
        is_technical_path = bool(
            DOI_PATTERN.fullmatch(first_decoded)
            or SCOPED_PACKAGE_PATTERN.fullmatch(first_decoded)
            or DOTTED_RUNTIME_PATH_PATTERN.fullmatch(first_decoded)
            or REVERSE_DOMAIN_NAMESPACE_PATTERN.fullmatch(first_decoded)
            or (
                "/" in first_decoded
                and first_authority.rsplit(".", 1)[-1].casefold() in TECHNICAL_FILE_SUFFIXES
                and not _has_public_suffix(first_authority)
            )
        )
        starts_query_clause = _looks_like_query_clause(candidate)
        contains_spaced_slash = bool(re.search(r"\s[\\/]\s", candidate))
        contains_spaced_date = bool(re.search(r"\b\d{4}\s*/\s*\d{1,2}\s*/\s*\d{1,2}\b", candidate))
        if (
            _is_local_path_token(first_token)
            or _parsed_url_token(first_token) is not None
            or is_technical_path
            or starts_query_clause
            or contains_spaced_slash
            or contains_spaced_date
        ):
            return start + boundary.start()
        if "/" in candidate or "\\" in candidate:
            # The delimiter is part of a legal path containing punctuation or
            # a conjunction, such as ``Research and Development/report.pdf``.
            continue
        if re.search(r"\.[a-z0-9]{1,12}[\]})>”’」』】）》]*$", candidate, re.IGNORECASE):
            # Commas and semicolons are legal inside a filename. Continue to
            # consume a terminal filename-shaped suffix rather than exposing it.
            continue
        return start + boundary.start()
    return line_end


def _replace_path_spans(value: str, pattern: re.Pattern[str]) -> str:
    pieces: list[str] = []
    cursor = 0
    while match := pattern.search(value, cursor):
        prefix = match.group("prefix")
        replacement_start = match.start() + len(prefix)
        path_start = match.start("path")
        path_end = _path_clause_end(value, path_start)
        if path_end <= path_start:
            cursor = match.end()
            continue
        pieces.append(value[cursor:replacement_start])
        pieces.append("local path")
        cursor = path_end
    pieces.append(value[cursor:])
    return "".join(pieces)


def _replace_labeled_path_spans(value: str) -> str:
    pieces: list[str] = []
    cursor = 0
    search_pos = 0
    while match := LABELED_LOCAL_PATH_START_PATTERN.search(value, search_pos):
        path_start = match.start("path")
        path_end = _path_clause_end(value, path_start)
        candidate = _privacy_decode(value[path_start:path_end])
        if candidate.startswith("//"):
            # This is an explicit file:// URI, not a relative value following
            # a human-readable `file:` label. Token-level URI handling consumes
            # the complete legal path, including comma/semicolon characters.
            search_pos = match.end()
            continue
        starts_query_clause = _looks_like_query_clause(candidate)
        if not _has_relative_path_evidence(candidate) or starts_query_clause:
            search_pos = match.end()
            continue
        prefix = match.group("prefix")
        replacement_start = match.start() + len(prefix)
        pieces.append(value[cursor:replacement_start])
        pieces.append("local path")
        cursor = path_end
        search_pos = path_end
    pieces.append(value[cursor:])
    return "".join(pieces)


def _url_replacement(token: str, parsed: ParseResult) -> str:
    stripped = token.strip().strip(WRAPPING_PUNCTUATION)
    normalized = _privacy_decode(_cleaned_token(token)).replace("\\", "/")
    explicit_http = normalized.casefold().startswith(("http://", "https://", "//"))
    forced_network = bool(URL_LABEL_PATTERN.match(stripped))
    hostname = parsed.hostname or ""
    if _is_private_network_host(hostname) or (
        (explicit_http or forced_network) and not _has_public_suffix(hostname)
    ):
        return "network resource"
    safe_hostname = re.sub(r"[^\w.-]+", " ", hostname, flags=re.UNICODE).strip()
    return safe_hostname or "public resource"


def _first_explicit_url_end(token: str) -> int:
    """Split compact URL lists without splitting legal URL punctuation.

    Commas and semicolons are valid URL-path/data characters, so they stay in
    the private payload by default.  The one unambiguous compact-list boundary
    is punctuation immediately followed by a fresh hierarchical URI.
    ``data:`` and ``file:`` values remain atomic even if their payload happens
    to contain URI-looking text.
    """

    normalized = _privacy_decode(_cleaned_token(token)).casefold()
    if normalized.startswith(("data:", "file:")):
        return len(token)
    boundary = FRESH_EXPLICIT_URL_PATTERN.search(token)
    return boundary.start() if boundary is not None else len(token)


def _replace_labeled_url_spans(value: str) -> str:
    pieces: list[str] = []
    cursor = 0
    search_pos = 0
    while match := LABELED_URL_START_PATTERN.search(value, search_pos):
        raw_token = match.group("url")
        token = raw_token[: _first_explicit_url_end(raw_token)]
        decoded = _privacy_decode(token)
        if not any(marker in decoded for marker in ("/", "\\", "?", "#", "@", ":")):
            search_pos = match.end()
            continue
        parsed = _parsed_url_token(token, force_authority=True)
        if parsed is None:
            search_pos = match.end()
            continue
        prefix = match.group("prefix")
        replacement_start = match.start() + len(prefix)
        token_without_separator = token.rstrip(",;")
        minimum_end = match.start("url") + len(token_without_separator)
        value_end = _path_clause_end(value, match.start("url"), minimum_end=minimum_end)
        pieces.append(value[cursor:replacement_start])
        pieces.append(_url_replacement(match.group(0), parsed))
        cursor = value_end
        search_pos = value_end
    pieces.append(value[cursor:])
    return "".join(pieces)


def _replace_detected_url_spans(value: str) -> str:
    pieces: list[str] = []
    cursor = 0
    search_pos = 0
    while match := DETECTED_URL_START_PATTERN.search(value, search_pos):
        raw_token = match.group("url")
        token = raw_token[: _first_explicit_url_end(raw_token)]
        parsed = _parsed_url_token(token)
        if parsed is None:
            search_pos = match.end()
            continue
        prefix = match.group("prefix")
        replacement_start = match.start() + len(prefix)
        token_without_separator = token.rstrip(",;")
        minimum_end = match.start("url") + len(token_without_separator)
        value_end = _path_clause_end(value, match.start("url"), minimum_end=minimum_end)
        pieces.append(value[cursor:replacement_start])
        pieces.append(_url_replacement(token, parsed))
        cursor = value_end
        search_pos = value_end
    pieces.append(value[cursor:])
    return "".join(pieces)


def _parsed_url_token(token: str, *, force_authority: bool = False) -> ParseResult | None:
    stripped = token.strip().strip(WRAPPING_PUNCTUATION)
    force_authority = force_authority or bool(URL_LABEL_PATTERN.match(stripped))
    cleaned = _cleaned_token(token)
    decoded = _privacy_decode(cleaned)
    normalized = decoded.replace("\\", "/")
    explicit_hierarchical_url = bool(
        normalized.startswith("//") or re.match(r"^[a-z][a-z0-9+.-]*://", normalized, re.IGNORECASE)
    )
    authority_guess = normalized.lstrip("/").split("/", 1)[0].split("@")[-1].split(":", 1)[0]
    reverse_namespace = bool(REVERSE_DOMAIN_NAMESPACE_PATTERN.fullmatch(normalized))
    reverse_labels = authority_guess.casefold().split(".")
    if reverse_namespace and len(reverse_labels) >= 2:
        reverse_namespace = reverse_labels[1] not in {
            "co",
            "com",
            "edu",
            "gov",
            "io",
            "net",
            "org",
        }
    if (
        not cleaned
        or normalized.casefold() in {"data:", "file:"}
        or DOI_PATTERN.fullmatch(normalized)
        or SCOPED_PACKAGE_PATTERN.fullmatch(normalized)
        or (DOTTED_RUNTIME_PATH_PATTERN.fullmatch(normalized) and not _is_ip_host(authority_guess))
        or reverse_namespace
        or (
            _is_local_path_token(token)
            and not (explicit_hierarchical_url and not normalized.casefold().startswith("file:"))
        )
    ):
        return None
    # Browsers and URL consumers commonly treat backslashes as path separators,
    # while urllib.parse otherwise leaves them inside netloc. Decode percent
    # escapes before parsing so encoded separators cannot become retained host
    # search terms.
    url_value = normalized
    scp_remote = SCP_REMOTE_PATTERN.fullmatch(url_value)
    scp_host_remote = SCP_HOST_REMOTE_PATTERN.fullmatch(url_value)
    scp_host_remote_valid = bool(
        scp_host_remote is not None
        and not scp_host_remote.group("path").isdigit()
        and (
            _has_public_suffix(scp_host_remote.group("host"))
            or _is_ip_host(scp_host_remote.group("host"))
        )
    )
    if scp_remote is not None:
        candidate = (
            f"ssh://{scp_remote.group('user')}@{scp_remote.group('host')}/"
            f"{scp_remote.group('path')}"
        )
    elif scp_host_remote_valid and scp_host_remote is not None:
        candidate = f"ssh://{scp_host_remote.group('host')}/{scp_host_remote.group('path')}"
    elif OPAQUE_URI_PATTERN.fullmatch(url_value):
        return urlparse(url_value)
    elif url_value.startswith("//"):
        candidate = f"https:{url_value}"
    elif EXPLICIT_SCHEME_PATTERN.match(url_value):
        candidate = url_value
    else:
        candidate = f"https://{url_value}"

    try:
        parsed = urlparse(candidate)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return None
    if (
        scp_remote is not None
        or scp_host_remote_valid
        or EXPLICIT_SCHEME_PATTERN.match(url_value)
        or url_value.startswith("//")
        or force_authority
    ):
        return parsed

    has_private_marker = any(marker in url_value for marker in ("?", "#", "@"))
    has_explicit_port = port is not None and not hostname.isdigit()
    suffix = hostname.rsplit(".", 1)[-1].casefold()
    has_public_suffix = _has_public_suffix(hostname)
    if (
        suffix in TECHNICAL_FILE_SUFFIXES
        and not has_public_suffix
        and not has_private_marker
        and not has_explicit_port
    ):
        # Natural package/file notation such as node.js/npm or
        # package.json/scripts is not a pasted public URL. Query, fragment,
        # credential, port, or genuine public-suffix markers still win in
        # favor of privacy (``example.md/private`` is a real URL shape).
        return None
    public_host_shape = (
        _is_ip_host(hostname) or _is_private_network_host(hostname) or has_public_suffix
    )
    if not (has_private_marker or has_explicit_port or public_host_shape):
        return None
    return parsed


def contains_url_token(value: str) -> bool:
    """Return whether text contains a URL-like token safe to route as web intent.

    DOI identifiers and local filesystem paths are deliberately excluded: they are
    scholarly identifiers or private local inputs, not an instruction to browse a
    public resource.
    """

    for match in NONSPACE_TOKEN_PATTERN.finditer(value):
        original = match.group(0)
        cleaned = _cleaned_token(original)
        if not cleaned or DOI_PATTERN.fullmatch(cleaned):
            continue
        if cleaned.startswith("//") or EXPLICIT_SCHEME_PATTERN.match(cleaned):
            if cleaned.casefold().startswith(("data:", "file:")):
                continue
            return True
        if _is_local_path_token(cleaned):
            continue
        if OPAQUE_URI_PATTERN.fullmatch(_privacy_decode(cleaned)):
            return True
        if cleaned.casefold().startswith("www."):
            return True

        forced_label = bool(URL_LABEL_PATTERN.match(original.strip(WRAPPING_PUNCTUATION)))
        parsed = _parsed_url_token(original, force_authority=forced_label)
        if forced_label and parsed is not None:
            return True
        hostname = parsed.hostname if parsed is not None else None
        if not hostname:
            continue
        # A bare dotted token only signals web intent when it also looks like a
        # navigable path/query. This avoids sending package.json or v1.2.3 tasks
        # outside the machine. Common source-file suffixes are excluded even when
        # followed by a slash (for example node.js/npm).
        has_navigation = bool(parsed.path not in ("", "/") or parsed.query or parsed.fragment)
        if has_navigation and (_has_public_suffix(hostname) or _is_ip_host(hostname)):
            return True
    return False


def contains_doi_token(value: str) -> bool:
    """Return whether text contains a DOI identifier."""

    return any(
        DOI_PATTERN.fullmatch(_privacy_decode(_cleaned_token(match.group(0)))) is not None
        for match in NONSPACE_TOKEN_PATTERN.finditer(value)
    )


def redact_url_tokens(value: str) -> str:
    """Replace pasted URLs with public hostnames before retrieval planning."""

    def replace(match: re.Match[str]) -> str:
        if _is_local_path_token(match.group(0)):
            return "local path"
        parsed = _parsed_url_token(match.group(0))
        if parsed is None:
            return match.group(0)
        return _url_replacement(match.group(0), parsed)

    def replace_wrapped(match: re.Match[str]) -> str:
        wrapped_value = _wrapped_value(match)
        path_redacted = _replace_labeled_path_spans(wrapped_value)
        path_redacted = _replace_path_spans(path_redacted, ROOTED_LOCAL_PATH_START_PATTERN)
        if path_redacted != wrapped_value:
            return path_redacted
        first_token = wrapped_value.strip().split(None, 1)[0] if wrapped_value.strip() else ""
        if match.group("markdown_destination") is not None and first_token:
            parsed = _parsed_url_token(first_token, force_authority=True)
            normalized_first = _privacy_decode(first_token).replace("\\", "/")
            if parsed is not None and (
                _has_public_suffix(parsed.hostname or "")
                or _is_ip_host(parsed.hostname or "")
                or _is_private_network_host(parsed.hostname or "")
                or normalized_first.startswith("//")
                or EXPLICIT_SCHEME_PATTERN.match(normalized_first)
            ):
                return _url_replacement(first_token, parsed)
            if any(marker in _privacy_decode(wrapped_value) for marker in ("/", "\\")):
                return "local path"
        if first_token:
            parsed = _parsed_url_token(first_token)
            if parsed is not None:
                return _url_replacement(first_token, parsed)
        redacted = NONSPACE_TOKEN_PATTERN.sub(replace, wrapped_value)
        return redacted if redacted != wrapped_value else match.group(0)

    wrapped_redacted = WRAPPED_SPAN_PATTERN.sub(replace_wrapped, value)
    url_redacted = _replace_labeled_url_spans(wrapped_redacted)
    detected_url_redacted = _replace_detected_url_spans(url_redacted)
    labeled_redacted = _replace_labeled_path_spans(detected_url_redacted)
    path_redacted = _replace_path_spans(labeled_redacted, ROOTED_LOCAL_PATH_START_PATTERN)
    return NONSPACE_TOKEN_PATTERN.sub(replace, path_redacted)


def private_url_terms(value: str) -> set[str]:
    """Extract non-host URL material that must never appear in provider queries."""

    def terms(text: str) -> set[str]:
        decoded = _privacy_decode(text)
        return {term.casefold() for term in re.findall(r"[^\W_]{3,}", decoded, flags=re.UNICODE)}

    # Using the exact service redaction as the authority keeps this replay guard
    # aligned with every supported wrapper/encoding. Terms retained in the safe
    # query (including a public hostname) are not treated as private.
    return terms(value) - terms(redact_url_tokens(value))


def queries_reveal_private_url_terms(queries: Iterable[str], source: str) -> bool:
    forbidden = private_url_terms(source)
    if not forbidden:
        return False
    return any(
        forbidden & set(re.findall(r"[^\W_]{3,}", query.casefold(), flags=re.UNICODE))
        for query in queries
    )
