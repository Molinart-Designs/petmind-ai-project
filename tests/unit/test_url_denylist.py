"""Unit tests for ``TRUSTED_EXTERNAL_DENYLIST`` parsing and matching."""

from pydantic import HttpUrl

from src.research.url_denylist import TrustedUrlDenylist


def test_host_rule_blocks_subdomains() -> None:
    d = TrustedUrlDenylist.from_raw("reddit.com")
    assert d.is_blocked(HttpUrl("https://www.reddit.com/r/dogs")) is True
    assert d.is_blocked(HttpUrl("https://old.reddit.com/x")) is True
    assert d.is_blocked(HttpUrl("https://avma.org/resources")) is False


def test_path_rule_blocks_prefix_only() -> None:
    d = TrustedUrlDenylist.from_raw("https://example.org/promotions/spam")
    assert d.is_blocked(HttpUrl("https://example.org/promotions/spam")) is True
    assert d.is_blocked(HttpUrl("https://www.example.org/promotions/spam/deeper")) is True
    assert d.is_blocked(HttpUrl("https://example.org/promotions/spamstuff")) is False
    assert d.is_blocked(HttpUrl("https://example.org/other")) is False


def test_bare_host_with_path_segment() -> None:
    d = TrustedUrlDenylist.from_raw("example.com/admin")
    assert d.is_blocked("https://sub.example.com/admin/login") is True
    assert d.is_blocked("https://example.com/") is False


def test_empty_denylist_never_blocks() -> None:
    d = TrustedUrlDenylist.from_raw("  ,  , ")
    assert d.is_blocked(HttpUrl("https://anything.test/x")) is False


def test_multiple_comma_separated_rules() -> None:
    d = TrustedUrlDenylist.from_raw("bad.example, https://good.example/blocked-path")
    assert d.is_blocked("https://sub.bad.example/hi") is True
    assert d.is_blocked("https://good.example/blocked-path/doc") is True
    assert d.is_blocked("https://good.example/safe") is False
