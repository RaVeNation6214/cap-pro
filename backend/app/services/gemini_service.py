"""
Gemini AI Service for Smart Contract Security Assistance.

Provides AI-powered explanations and fix suggestions for detected vulnerabilities.
Requires GEMINI_API_KEY environment variable.
"""
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Try to import google-genai (new SDK)
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-genai not installed. AI assistant unavailable.\n"
        "Install with: pip install google-genai"
    )


SYSTEM_PROMPT = """You are an expert Solidity smart contract security auditor with
deep knowledge of the Ethereum blockchain, DeFi protocols, and common attack vectors.
You help developers understand and fix smart contract vulnerabilities.

When analyzing code:
1. Be precise and technical
2. Always show the vulnerable code pattern
3. Provide a concrete fixed version
4. Explain WHY it's vulnerable
5. Keep responses concise but complete

Format your response as:
## Vulnerability Analysis
[Brief explanation]

## Vulnerable Pattern
```solidity
[vulnerable code snippet]
```

## Fixed Code
```solidity
[corrected code snippet]
```

## Additional Notes
[Any extra security considerations]
"""


class GeminiService:
    """
    AI assistant powered by Google Gemini for vulnerability explanations and fixes.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.available = False
        self.model = None

        if not GEMINI_AVAILABLE:
            return

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            logger.warning(
                "GEMINI_API_KEY not set. AI assistant will be unavailable.\n"
                "Set it with: export GEMINI_API_KEY=your_key"
            )
            return

        try:
            self.client = genai.Client(api_key=key)
            self.model_name = "gemini-2.0-flash"
            self.available = True
            logger.info("Gemini AI service initialized successfully (google-genai SDK).")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")

    def get_help(self, code: str, issue: str) -> Dict[str, str]:
        """
        Get AI-powered explanation and fix for a detected vulnerability.

        Args:
            code: Solidity contract source code
            issue: Vulnerability type (e.g., 'reentrancy', 'access_control')

        Returns:
            Dict with 'response' (markdown string) and 'status'
        """
        if not self.available:
            return {
                "response": self._fallback_response(issue),
                "status": "fallback",
                "model": "static",
            }

        try:
            prompt = self._build_prompt(code, issue)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                ),
            )
            return {
                "response": response.text,
                "status": "success",
                "model": self.model_name,
            }
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            return {
                "response": self._fallback_response(issue),
                "status": "error",
                "model": "static",
                "error": str(e),
            }

    def _build_prompt(self, code: str, issue: str) -> str:
        """Build a structured prompt for vulnerability analysis."""
        issue_descriptions = {
            "reentrancy": "reentrancy (external call before state update)",
            "access_control": "access control vulnerability (tx.origin usage or missing authorization)",
            "arithmetic": "integer overflow/underflow vulnerability",
            "unchecked_calls": "unchecked external call return values",
        }

        issue_desc = issue_descriptions.get(
            issue.lower().replace(' ', '_').replace('-', '_'),
            issue
        )

        return f"""Analyze this Solidity smart contract for a detected {issue_desc} vulnerability.

Provide a detailed security analysis with:
1. What exactly is vulnerable in this code
2. How an attacker could exploit it
3. The corrected version of the vulnerable function(s)
4. Best practices to prevent this in the future

Contract code:
```solidity
{code[:3000]}
```
"""

    def _fallback_response(self, issue: str) -> str:
        """Static fallback when Gemini is unavailable."""
        from ..utils.suggestions import get_suggestion
        suggestion = get_suggestion(issue)

        return f"""## Vulnerability Analysis

**Type:** {issue.replace('_', ' ').title()}
**Severity:** {suggestion.get('severity', 'Unknown')}

## What Was Detected

{suggestion.get('explanation', 'A vulnerability was detected in this contract.')}

## How to Fix

{suggestion.get('suggestion', 'Review and update the vulnerable code.')}

## Example Fix

```solidity
{suggestion.get('fix_example', '// See documentation for fix patterns')}
```

## References

{chr(10).join('- ' + r for r in suggestion.get('references', ['https://swcregistry.io/']))}

---
*Note: Enable GEMINI_API_KEY for AI-powered personalized analysis.*
"""

    def is_available(self) -> bool:
        return self.available
