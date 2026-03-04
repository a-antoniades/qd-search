"""AI-powered PR review using Claude API.

Posts review comments on PRs with focus on security, performance, or correctness.
Non-blocking: always exits 0 so CI isn't gated on review comments.
"""

import os
import subprocess
import sys

import anthropic

FOCUS_PROMPTS = {
    "security": (
        "Review this diff for security issues: injection vulnerabilities, "
        "unsafe deserialization, hardcoded secrets, path traversal, "
        "insecure randomness, and other OWASP Top 10 concerns."
    ),
    "performance": (
        "Review this diff for performance issues: unnecessary allocations, "
        "O(n^2) algorithms where O(n) is possible, missing caching opportunities, "
        "redundant computation, and memory leaks."
    ),
    "correctness": (
        "Review this diff for correctness issues: off-by-one errors, "
        "race conditions, unhandled edge cases, incorrect type assumptions, "
        "broken invariants, and logic errors."
    ),
}


def main():
    diff_path = sys.argv[1]
    focus = os.environ["FOCUS"]
    pr_number = os.environ["PR_NUMBER"]
    repo = os.environ["REPO"]

    with open(diff_path) as f:
        diff = f.read()

    if not diff.strip():
        print("Empty diff, skipping review.")
        return

    # Bound context to avoid excessive token usage
    if len(diff) > 50000:
        diff = diff[:50000] + "\n\n... (diff truncated at 50k chars)"

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a code reviewer focused on **{focus}**.\n\n"
                    f"{FOCUS_PROMPTS[focus]}\n\n"
                    "If you find issues, list them concisely with file paths and line numbers. "
                    "If the code looks good, say so briefly. "
                    "Do not repeat the diff back. Be specific and actionable.\n\n"
                    f"```diff\n{diff}\n```"
                ),
            }
        ],
    )

    review_text = response.content[0].text
    comment_body = f"## AI Review: {focus.title()}\n\n{review_text}\n\n---\n*Automated review by Claude*"

    # Post as PR comment via gh CLI
    subprocess.run(
        [
            "gh", "api",
            f"repos/{repo}/issues/{pr_number}/comments",
            "-f", f"body={comment_body}",
        ],
        check=False,  # Non-blocking
    )
    print(f"Posted {focus} review comment on PR #{pr_number}")


if __name__ == "__main__":
    main()
