from __future__ import annotations

from localllm.query_privacy import (
    contains_doi_token,
    contains_url_token,
    private_url_terms,
    redact_url_tokens,
)


def test_non_url_slashes_times_dates_and_dois_keep_their_search_meaning() -> None:
    value = (
        "Compare TCP/IP and quality/speed on 2026/08/09 at 10:30 using "
        "DOI 10.1038/s41586-024-07487-w"
    )

    assert redact_url_tokens(value) == value


def test_local_paths_are_not_forwarded_as_retrieval_terms() -> None:
    value = r"Inspect C:\Users\Alice\secret.txt and /home/alice/private.log"
    redacted = redact_url_tokens(value)

    assert "Alice" not in redacted
    assert "alice" not in redacted
    assert "secret" not in redacted
    assert "private" not in redacted
    assert redacted.count("local path") >= 1


def test_generic_uri_schemes_retain_only_the_public_authority() -> None:
    redacted = redact_url_tokens("Verify s3://private-bucket/SECRETKEY now")

    assert "private-bucket" in redacted
    assert "SECRETKEY" not in redacted


def test_url_intent_detection_excludes_dois_and_local_paths() -> None:
    assert contains_url_token("Open example.org/reference")
    assert contains_url_token("Open //example.org/reference")
    assert contains_url_token("Inspect s3://public-bucket/object")
    assert not contains_url_token("DOI 10.1038/s41586-024-07487-w")
    assert not contains_url_token(r"Inspect C:\\Users\\Alice\\report.txt")
    assert not contains_url_token("Inspect /home/alice/report.txt")
    assert contains_doi_token("DOI 10.1038/s41586-024-07487-w")


def test_explicit_ipv6_urls_remain_web_signals_not_local_paths() -> None:
    values = [
        "https://[2606:4700:4700::1111]/private",
        "//[2001:4860:4860::8888]/private",
    ]

    for value in values:
        assert contains_url_token(value)
        assert "private" not in redact_url_tokens(value)


def test_url_intent_detection_does_not_mistake_technical_tokens_for_navigation() -> None:
    assert not contains_url_token("Write package.json")
    assert not contains_url_token("Compare v1.2.3 with v1.3.0")
    assert not contains_url_token("Explain node.js/npm and TCP/IP")


def test_technical_package_paths_survive_retrieval_redaction() -> None:
    value = (
        "Search node.js/npm, package.json/scripts, @modelcontextprotocol/sdk, "
        "@openai/codex, v1.2.3/changelog, and python3.12/asyncio compatibility"
    )

    assert redact_url_tokens(value) == value
    assert private_url_terms(value) == set()


def test_dotted_code_namespaces_stay_intact_but_public_hosts_are_recognized() -> None:
    technical = (
        "fastapi.middleware/cors torch.nn/functional os.path/join "
        "react.dom/render com.example/module com.google/android/gms "
        "com.apple/foundation com.microsoft/graph org.apache/http"
    )

    assert redact_url_tokens(technical) == technical
    assert private_url_terms(technical) == set()
    assert not contains_url_token(technical)
    assert contains_url_token("docs.python.org/3/library/asyncio.html")


def test_embedded_urls_and_labeled_paths_are_redacted() -> None:
    value = (
        "Search URL:https://example.com/private?token=TOPSECRET and "
        "[this](https://example.org/hidden?sig=SIGNED) with "
        "path:/home/alice/secret.txt file=C:\\Users\\Alice\\private.log "
        "and `/srv/internal/notes.txt`"
    )
    redacted = redact_url_tokens(value)

    assert "example.com" in redacted
    assert "example.org" in redacted
    assert "TOPSECRET" not in redacted
    assert "SIGNED" not in redacted
    assert "alice" not in redacted.casefold()
    assert "secret" not in redacted.casefold()
    assert "private" not in redacted.casefold()
    assert redacted.count("local path") >= 1


def test_generic_labels_json_and_quoted_space_paths_are_redacted_as_whole_values() -> None:
    values = [
        "cwd=/home/alice/secret.txt",
        "dir:/home/alice/secret.txt",
        "directory=/home/alice/secret.txt",
        "location:/home/alice/secret.txt",
        '{"path":"/home/alice/secret.txt"}',
        r'{"path":"C:\Users\Alice\secret.txt"}',
        '"/home/alice/My Project/secret plan.txt"',
        r'"C:\Users\Alice\My Project\secret plan.txt"',
        "`/home/alice/My Project/secret plan.txt`",
    ]

    for value in values:
        redacted = redact_url_tokens(f"Search {value}")
        assert "local path" in redacted
        assert "alice" not in redacted.casefold()
        assert "secret" not in redacted.casefold()
        assert "project" not in redacted.casefold()
        assert {"alice", "secret"} <= private_url_terms(value)


def test_json_wrapped_bare_and_scheme_relative_urls_drop_private_components() -> None:
    for value in [
        '{"url":"example.org/private?token=TOPSECRET"}',
        '{"url":"//example.org/private?token=TOPSECRET"}',
    ]:
        redacted = redact_url_tokens(f"Search {value}")
        assert "example.org" in redacted
        assert "private" not in redacted
        assert "TOPSECRET" not in redacted
        assert {"private", "topsecret"} <= private_url_terms(value)


def test_full_width_cjk_labels_do_not_expose_paths_or_signed_urls() -> None:
    values = [
        "搜索 路径：/home/alice/secret.txt",
        r"搜索 路径：C:\Users\Alice\secret.txt",
        "搜索 链接：example.com/private?token=TOPSECRET",
    ]

    for value in values:
        redacted = redact_url_tokens(value)
        assert "alice" not in redacted.casefold()
        assert "secret" not in redacted.casefold()
        assert "private" not in redacted.casefold()
        assert "TOPSECRET" not in redacted


def test_unquoted_and_fenced_paths_with_spaces_are_consumed_to_the_line_boundary() -> None:
    values = [
        r"Search C:\Users\Alice\My Project\secret plan.txt",
        "搜索 路径：/home/小明/秘密 项目/计划.txt",
        "Search ```\n/home/alice/My Project/secret plan.txt\n```",
        "Search ```text\nC:\\Users\\Alice\\My Project\\secret plan.txt\n```",
    ]

    for value in values:
        redacted = redact_url_tokens(value)
        assert "local path" in redacted
        for private_term in ("alice", "project", "secret", "小明", "秘密", "项目", "计划"):
            assert private_term not in redacted.casefold()


def test_spaced_slash_prose_division_and_dates_keep_their_meaning() -> None:
    values = [
        "Compare input / output quality",
        "Calculate 10 / 2 and explain",
        "Search pros / cons of local LLMs",
        "Find 2026 / 08 / 09 release notes",
    ]

    for value in values:
        assert redact_url_tokens(value) == value


def test_unquoted_path_redaction_preserves_following_query_clauses() -> None:
    value = (
        "Search /home/alice/secret.txt and Python documentation with "
        r"C:\Users\Alice\private.txt then compare results"
    )

    assert redact_url_tokens(value) == (
        "Search local path and Python documentation with local path then compare results"
    )


def test_arbitrary_plain_labels_before_bare_signed_urls_are_redacted() -> None:
    for label in ("ref", "target", "host", "anything"):
        value = f"Search {label}:example.com/private?token=TOPSECRET"
        redacted = redact_url_tokens(value)
        assert "example.com" in redacted
        assert "private" not in redacted
        assert "TOPSECRET" not in redacted
        assert {"private", "topsecret"} <= private_url_terms(value)


def test_backslash_and_encoded_url_paths_never_become_public_host_terms() -> None:
    values = [
        r"Verify https://example.com\private\SECRET?token=TOPSECRET",
        r"Verify example.com\private\SECRET?token=TOPSECRET",
        "Verify https://example.com%2Fprivate%2FSECRET?token=TOPSECRET",
        "Verify example.com%2Fprivate%2FSECRET?token=TOPSECRET",
        r"Verify https://example.com:443\private\SECRET?token=TOPSECRET",
        "Verify https://example.com%5Cprivate%5CSECRET?token=TOPSECRET",
    ]

    for value in values:
        redacted = redact_url_tokens(value)
        assert redacted == "Verify example.com"
        assert {"private", "secret", "token", "topsecret"} <= private_url_terms(value)


def test_encoded_and_file_uri_local_paths_are_fully_suppressed() -> None:
    values = [
        "%2Fhome%2Falice%2Fsecret.txt",
        "path:%2Fhome%2Falice%2Fsecret.txt",
        "C%3A%5CUsers%5CAlice%5Csecret.txt",
        "%5C%5Cserver%5Cshare%5CAlice%5Csecret.txt",
        "file:%2Fhome%2Falice%2Fsecret.txt",
        "file:///home/alice/secret.txt",
        "file://server/share/alice/secret.txt",
    ]

    for value in values:
        assert redact_url_tokens(f"Search {value}") == "Search local path"
        assert {"alice", "secret"} <= private_url_terms(value)


def test_encoded_scholarly_and_code_identifiers_keep_their_exact_search_text() -> None:
    values = [
        "10.1038%2Fs41586-024-07487-w",
        "%40openai%2Fcodex",
        "com.google%2Fandroid%2Fgms",
    ]

    for value in values:
        assert redact_url_tokens(value) == value
        assert private_url_terms(value) == set()


def test_private_and_scheme_relative_hosts_drop_every_path_component() -> None:
    values = [
        "//192.168.1.7/private/TOPSECRET",
        "//intranet/private/TOPSECRET",
        "192.168.1.7/private/TOPSECRET",
        "intranet/private/TOPSECRET",
    ]

    for value in values:
        redacted = redact_url_tokens(f"Verify {value} now")
        assert redacted == "Verify network resource now"
        assert {"private", "topsecret"} <= private_url_terms(value)


def test_smart_markdown_and_cjk_wrapped_paths_are_redacted_as_whole_values() -> None:
    values = [
        "“/home/alice/My Project/TOPSECRET plan.txt”",
        "「/home/alice/My Project/TOPSECRET plan.txt」",
        "**/home/alice/My Project/TOPSECRET plan.txt**",
        "_/home/alice/My Project/TOPSECRET plan.txt_",
        "~~/home/alice/My Project/TOPSECRET plan.txt~~",
        "</home/alice/My Project/TOPSECRET plan.txt>",
        "[path](/home/alice/My Project/TOPSECRET plan.txt)",
    ]

    for value in values:
        assert redact_url_tokens(f"Search {value} latest") == "Search local path latest"
        assert {"alice", "project", "topsecret"} <= private_url_terms(value)


def test_relative_environment_and_explicitly_labeled_paths_are_private() -> None:
    values = [
        r".\private\TOPSECRET.txt",
        r"..\private\TOPSECRET.txt",
        "$HOME/Projects/Secret Client/plan.txt",
        "${HOME}/Projects/Secret Client/plan.txt",
        r"%USERPROFILE%\Documents\Secret Client\plan.txt",
        "path:src/private/TOPSECRET.txt",
        "file=secrets/TOPSECRET.json",
        "cwd:Projects/Secret Client/plan.txt",
        r"directory=private\TOPSECRET.txt",
        "路径：项目/秘密/计划.txt",
    ]

    for value in values:
        redacted = redact_url_tokens(f"Search {value} and Python documentation")
        assert redacted == "Search local path and Python documentation"
        assert "TOPSECRET" not in redacted


def test_path_punctuation_inside_legal_names_does_not_leak_a_suffix() -> None:
    values = [
        "/home/alice/Research and Development/TOPSECRET",
        "/home/alice/My, Private Project/TOPSECRET",
    ]

    for value in values:
        assert redact_url_tokens(f"Search {value} and Python docs") == (
            "Search local path and Python docs"
        )
        assert {"alice", "topsecret"} <= private_url_terms(value)


def test_public_suffixes_win_over_source_file_suffix_heuristics() -> None:
    for hostname in ("example.md", "example.cc", "example.rs", "example.sh"):
        value = f"Search {hostname}/private/TOPSECRET"
        assert redact_url_tokens(value) == f"Search {hostname}"
        assert contains_url_token(value)
        assert {"private", "topsecret"} <= private_url_terms(value)


def test_worldwide_wrappers_and_balanced_markdown_destinations_are_private() -> None:
    values = [
        "【/home/alice/My Project/TOPSECRET plan.txt】",
        "（/home/alice/My Project/TOPSECRET plan.txt）",
        "《/home/alice/My Project/TOPSECRET plan.txt》",
        "«/home/alice/My Project/TOPSECRET plan.txt»",
        "‹/home/alice/My Project/TOPSECRET plan.txt›",
        "„/home/alice/My Project/TOPSECRET plan.txt“",
        "｢/home/alice/My Project/TOPSECRET plan.txt｣",
        "〈/home/alice/My Project/TOPSECRET plan.txt〉",
        "〔/home/alice/My Project/TOPSECRET plan.txt〕",
        "［/home/alice/My Project/TOPSECRET plan.txt］",
        "[path](/home/alice/My (Project) TOPSECRET/file.txt)",
    ]

    for value in values:
        assert redact_url_tokens(f"Verify {value}") == "Verify local path"
        assert {"alice", "topsecret"} <= private_url_terms(value)


def test_explicit_url_labels_and_spaced_url_paths_keep_no_private_suffix() -> None:
    values = [
        "URL:corpserver/private/TOPSECRET",
        "host:myserver/private/TOPSECRET",
        "source:buildhost/private/TOPSECRET",
        '"https://example.com/private Project/TOPSECRET plan"',
        "URL:https://example.com/private Project/TOPSECRET plan",
        "https://example.com/private Project/TOPSECRET plan",
        "//devbox/private Project/TOPSECRET plan",
    ]

    for value in values:
        redacted = redact_url_tokens(value)
        assert "TOPSECRET" not in redacted
        assert {"private", "topsecret"} <= private_url_terms(value)

    assert contains_url_token("URL:corpserver/private/TOPSECRET")


def test_path_boundaries_preserve_real_followup_queries_but_not_filename_suffixes() -> None:
    followups = [
        "node.js/npm compatibility",
        "10.1038/s41586-024-07487-w evidence",
        "compare quality/speed",
        "compare 2026/08/09 releases",
        "explain TCP/IP performance",
        "research @openai/codex docs",
        "with input / output quality",
        "calculate 10 / 2",
    ]
    for followup in followups:
        separator = " " if followup.startswith("with ") else " and "
        assert redact_url_tokens(f"/home/alice/secret.txt{separator}{followup}") == (
            f"local path{separator}{followup}"
        )

    for path in (
        "/home/alice/Report, final TOPSECRET.txt",
        "/home/alice/Report; final TOPSECRET.txt",
    ):
        assert redact_url_tokens(path) == "local path"


def test_labels_without_path_evidence_are_not_misclassified() -> None:
    values = [
        "location: Hong Kong latest weather",
        "path: explain os.path/join behavior",
        "directory: compare node.js/npm versions",
    ]

    for value in values:
        assert redact_url_tokens(value) == value
        assert private_url_terms(value) == set()


def test_nested_encoding_ip_hosts_scp_remotes_and_named_home_paths_are_safe() -> None:
    values_and_safe = [
        ("%252Fhome%252Falice%252FTOPSECRET.txt", "local path"),
        ("example.com%252Fprivate%252FTOPSECRET?token=SIGNED", "example.com"),
        ("192.168.1.7/TOPSECRET", "network resource"),
        ("8.8.8.8/TOPSECRET", "8.8.8.8"),
        ("dev.to/private/TOPSECRET", "dev.to"),
        ("org.com/private/TOPSECRET", "org.com"),
        ("git@example.org:private/TOPSECRET.git", "example.org"),
        ("~alice/private/TOPSECRET.txt", "local path"),
    ]

    for value, safe in values_and_safe:
        assert redact_url_tokens(value) == safe
        assert "topsecret" in private_url_terms(value)


def test_assignment_prefixed_spaced_paths_are_fully_redacted() -> None:
    for value in (
        "input=/home/alice/My Project/TOPSECRET.txt",
        "binary:/home/alice/My Project/TOPSECRET.txt",
        "--input=/home/alice/My Project/TOPSECRET.txt",
        "foo：/home/alice/My Project/TOPSECRET.txt",
    ):
        redacted = redact_url_tokens(value)
        assert "TOPSECRET" not in redacted
        assert "local path" in redacted


def test_labeled_relative_paths_may_contain_spaces_before_the_first_separator() -> None:
    values = [
        "path:My Private Project/TOPSECRET.txt",
        "cwd:My Private Project/TOPSECRET.txt",
        "file=My Private Project/TOPSECRET.txt",
        "directory:My Private Project/TOPSECRET.txt",
        "路径：我的 私人项目/TOPSECRET.txt",
        "“path:My Private Project/TOPSECRET.txt”",
    ]

    for value in values:
        assert redact_url_tokens(f"Search {value} and Python docs") == (
            "Search local path and Python docs"
        )
        assert "topsecret" in private_url_terms(value)


def test_multilingual_followup_clauses_survive_private_path_redaction() -> None:
    values_and_expected = [
        (
            "搜索 /home/小明/秘密.txt 和 比较 TCP/IP 性能",
            "搜索 local path 和 比较 TCP/IP 性能",
        ),
        (
            "搜索 /home/小明/秘密.txt 然后 研究 @openai/codex 文档",
            "搜索 local path 然后 研究 @openai/codex 文档",
        ),
        (
            r"搜索 C:\Users\小明\秘密.txt，比较 TCP/IP 性能",
            "搜索 local path，比较 TCP/IP 性能",
        ),
        (
            "検索 /home/taro/secret.txt と TCP/IP を比較",
            "検索 local path と TCP/IP を比較",
        ),
        (
            "검색 /home/minsu/secret.txt 그리고 TCP/IP 비교",
            "검색 local path 그리고 TCP/IP 비교",
        ),
    ]

    for value, expected in values_and_expected:
        assert redact_url_tokens(value) == expected


def test_opaque_uris_and_scp_host_remotes_never_forward_their_payload() -> None:
    values_and_safe = [
        ("sms:+85212345678?body=ZEBRASECRET", "public resource"),
        ("tel:+85212345678", "public resource"),
        ("urn:uuid:ZEBRASECRET", "public resource"),
        ("did:key:ZEBRASECRET", "public resource"),
        ("bitcoin:ZEBRASECRET", "public resource"),
        ("ethereum:ZEBRASECRET", "public resource"),
        ("github.com:private/ZEBRASECRET.git", "github.com"),
        ("github.com:ZEBRASECRET.git", "github.com"),
        ("gitlab.com:ZEBRASECRET", "gitlab.com"),
        ("192.0.2.1:ZEBRASECRET.git", "network resource"),
    ]

    for value, safe in values_and_safe:
        assert redact_url_tokens(value) == safe
        assert private_url_terms(value)
        assert contains_url_token(value)


def test_legal_url_punctuation_never_splits_a_private_path_or_data_payload() -> None:
    values_and_safe = [
        ("https://example.com/private,ZEBRASECRET", "example.com"),
        ("https://example.com/private;ZEBRASECRET", "example.com"),
        ("//example.com/private,ZEBRASECRET", "example.com"),
        ("s3://bucket/private,ZEBRASECRET", "bucket"),
        ("ssh://host/private,ZEBRASECRET", "host"),
        ("https://[2606:4700:4700::1111]/private,ZEBRASECRET", "network resource"),
        ("//[2001:4860:4860::8888]/private;ZEBRASECRET", "network resource"),
        ("file:///home/alice/ZEBRASECRET,tail", "local path"),
        ("file:///home/alice/ZEBRASECRET;tail", "local path"),
        ("data:text/plain,ZEBRASECRET", "public resource"),
        ("data:text/plain;charset=utf-8,ZEBRASECRET", "public resource"),
        ("data:text/plain;base64,WkVCUkFTRUNSRVQ=", "public resource"),
        ('data:application/json,{"token":"ZEBRASECRET"}', "public resource"),
    ]

    for value, safe in values_and_safe:
        assert redact_url_tokens(value) == safe
        assert private_url_terms(value)


def test_url_punctuation_preserves_followup_queries_and_compact_url_lists() -> None:
    values_and_safe = [
        (
            "Search https://example.com/private, then compare TCP/IP performance",
            "Search example.com then compare TCP/IP performance",
        ),
        (
            "Search https://example.com/private; then research @openai/codex docs",
            "Search example.com then research @openai/codex docs",
        ),
        (
            "Compare https://example.org/ZEBRASECRET,https://example.net/ZEBRASECRET",
            "Compare example.org example.net",
        ),
        (
            "Compare https://example.org/ZEBRASECRET;https://example.net/ZEBRASECRET",
            "Compare example.org example.net",
        ),
    ]

    for value, safe in values_and_safe:
        assert redact_url_tokens(value) == safe
        assert "zebrasecret" not in redact_url_tokens(value).casefold()


def test_topical_slash_labels_keep_their_general_search_meaning() -> None:
    values = [
        "location: Hong Kong/Kowloon latest weather",
        "location: Hong Kong / Kowloon weather",
        "位置：香港/九龙 今日天气",
        "file: PDF/EPUB comparison",
        "path: pros/cons of local LLMs",
        "directory: src/lib architecture",
        "dir: north/south directions",
    ]

    for value in values:
        assert redact_url_tokens(value) == value
        assert private_url_terms(value) == set()
