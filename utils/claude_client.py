"""Shared Anthropic client setup."""

import anthropic

from config.settings import settings


def get_claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)
