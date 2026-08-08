from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_ROOT / ".local" / "runtime" / "browser" / "evidence"
CDP_URL = "http://127.0.0.1:9470"
APP_URL = "http://127.0.0.1:8008/"


def run() -> dict[str, object]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    screenshots: list[str] = []
    checks: dict[str, object] = {}
    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(CDP_URL)
        context = browser.contexts[0]
        pages = [candidate for candidate in context.pages if candidate.url.startswith(APP_URL)]
        page = pages[0] if pages else context.new_page()
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.bring_to_front()
        page.goto(APP_URL, wait_until="networkidle")
        workspace = page.get_by_test_id("workspace")
        workspace.wait_for(state="visible")
        checks["title"] = page.title()
        checks["workspaceStatus"] = workspace.get_attribute("data-status")
        checks["initialView"] = workspace.get_attribute("data-view")

        home = EVIDENCE_DIR / f"{timestamp}-playground.png"
        page.screenshot(path=str(home), full_page=True)
        screenshots.append(str(home.relative_to(PROJECT_ROOT)))

        page.get_by_test_id("nav-models").click()
        page.locator('[data-view="models"]').wait_for(state="visible")
        page.wait_for_timeout(300)
        checks["modelCards"] = page.locator(".model-card").count()
        models = EVIDENCE_DIR / f"{timestamp}-models.png"
        page.screenshot(path=str(models), full_page=True)
        screenshots.append(str(models.relative_to(PROJECT_ROOT)))

        page.get_by_test_id("nav-api").click()
        page.locator('[data-view="api"]').wait_for(state="visible")
        page.wait_for_timeout(300)
        checks["apiCodeCards"] = page.locator(".code-card").count()
        api_page = EVIDENCE_DIR / f"{timestamp}-api.png"
        page.screenshot(path=str(api_page), full_page=True)
        screenshots.append(str(api_page.relative_to(PROJECT_ROOT)))

        page.get_by_test_id("nav-chat").click()
        chat_input = page.get_by_test_id("chat-input")
        chat_input.fill("A visible browser smoke test — do not send")
        checks["sendEnabledAfterTyping"] = page.get_by_test_id("chat-send").is_enabled()

        cdp = context.new_cdp_session(page)
        cdp.send(
            "Emulation.setDeviceMetricsOverride",
            {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True},
        )
        page.reload(wait_until="networkidle")
        page.get_by_test_id("workspace").wait_for(state="visible")
        mobile = EVIDENCE_DIR / f"{timestamp}-mobile.png"
        page.screenshot(path=str(mobile), full_page=True)
        screenshots.append(str(mobile.relative_to(PROJECT_ROOT)))
        checks["mobileScrollWidth"] = page.evaluate("document.documentElement.scrollWidth")
        checks["mobileClientWidth"] = page.evaluate("document.documentElement.clientWidth")
        cdp.send("Emulation.clearDeviceMetricsOverride")
        page.reload(wait_until="networkidle")
        page.bring_to_front()
        browser.close()

    failures: list[str] = []
    expected = {
        "title": "LocalLLM Studio",
        "workspaceStatus": "ready",
        "initialView": "chat",
        "modelCards": 9,
        "apiCodeCards": 3,
        "sendEnabledAfterTyping": True,
    }
    for name, value in expected.items():
        if checks.get(name) != value:
            failures.append(f"{name}: expected {value!r}, got {checks.get(name)!r}")
    if checks.get("mobileScrollWidth") != checks.get("mobileClientWidth"):
        failures.append(
            "mobile viewport overflows horizontally: "
            f"{checks.get('mobileScrollWidth')} > {checks.get('mobileClientWidth')}"
        )
    if console_errors:
        failures.append(f"browser console errors: {console_errors}")
    if page_errors:
        failures.append(f"uncaught page errors: {page_errors}")

    result = {
        "url": APP_URL,
        "title": checks.get("title"),
        "status": "failed" if failures else "passed",
        "checks": checks,
        "failures": failures,
        "screenshots": screenshots,
    }
    status_path = EVIDENCE_DIR / f"{timestamp}-status.json"
    status_path.write_text(json.dumps(result, indent=2))
    result["statusFile"] = str(status_path.relative_to(PROJECT_ROOT))
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(outcome, indent=2))
    raise SystemExit(1 if outcome["status"] == "failed" else 0)
