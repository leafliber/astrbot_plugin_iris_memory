import pytest

from iris_memory.config import Config


@pytest.mark.parametrize("allow_container_host", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080",
        "http://127.0.0.1/",
        "https://localhost",
        "https://localhost:443/",
        "http://[::1]:8080",
        "https://[::1]/",
        "HTTP://LOCALHOST:65535",
        "http://localhost:1",
    ],
)
def test_loopback_origins_remain_supported(url, allow_container_host):
    config = Config.from_mapping({"base_url": url, "allow_container_host": allow_container_host})
    assert config.base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://iris:8080",
        "https://iris",
        "http://iris-core:8080/",
        "http://iris.internal:8080",
        "http://iris-1.default:1",
        "HTTPS://IRIS:65535",
        "http://a",
        "http://1iris:80",
        "http://" + "a" * 63,
        "http://" + ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 61]),
    ],
)
def test_container_hostname_requires_explicit_opt_in(url):
    with pytest.raises(ValueError, match="allowed HTTP"):
        Config.from_mapping({"base_url": url})
    config = Config.from_mapping({"base_url": url, "allow_container_host": True})
    assert config.base_url == url and config.allow_container_host is True


@pytest.mark.parametrize("allow_container_host", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8:8080",
        "http://10.0.0.1:8080",
        "http://192.168.1.2",
        "http://0.0.0.0:8080",
        "http://127.0.0.2",
        "http://127.1",
        "http://2130706433",
        "http://0177.0.0.1",
        "http://0x7f000001",
        "http://0x7f.0.0.1",
        "http://0x08080808",
        "http://iris.123",
        "http://iris.0xff",
        "http://[2001:db8::1]:8080",
        "http://[::ffff:127.0.0.1]",
        "http://[::1%25lo0]",
        "http://[::1]extra:8080",
        "http://::1:8080",
        "http://localhost:0",
        "http://localhost:65536",
        "http://localhost:-1",
        "http://localhost:+80",
        "http://localhost:8080:8081",
        "http://localhost:",
        "http://localhost:port",
        "http://localhost:８０",
        "http://iris/api/v1",
        "http://localhost//",
        "http://iris/../",
        "http://iris?",
        "http://localhost?token=iris_test_token",
        "http://iris#",
        "http://localhost/#fragment",
        "http://iris_test_user:iris_test_token@iris:8080",
        "http://iris_test_token@localhost",
        "http://@localhost",
        "http://:iris_test_token@localhost",
        "http://localhost@iris",
        "http://localhost,iris",
        "http://iris;other",
        "http://iris\\other",
        "http://%69ris:8080",
        "http://iris_core:8080",
        "http://-iris:8080",
        "http://iris-:8080",
        "http://.iris:8080",
        "http://iris.:8080",
        "http://iris..internal:8080",
        "http://*.iris:8080",
        "http://鸢尾:8080",
        "http://Kiris:8080",
        "http://" + "a" * 64,
        "http://" + ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 62]),
        " http://localhost:8080",
        "http://localhost:8080 ",
        "http://local\thost:8080",
        "http://localhost:8080\n",
        "\x00http://localhost:8080",
        "http://iris\x7f:8080",
        "file:///tmp/iris",
        "ftp://iris:8080",
        "//iris:8080",
        "http:iris:8080",
        "http:///iris:8080",
        "iris:8080",
        "",
        None,
        8080,
    ],
)
def test_rejects_non_origin_and_ambiguous_host_forms(url, allow_container_host):
    with pytest.raises(ValueError, match="allowed HTTP"):
        Config.from_mapping({"base_url": url, "allow_container_host": allow_container_host})


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_container_switch_requires_boolean(value):
    with pytest.raises(ValueError, match="booleans"):
        Config.from_mapping({"allow_container_host": value})


def test_container_mode_still_reads_token_only_from_config(monkeypatch):
    monkeypatch.setenv("IRIS_HOST_TOKEN", "iris_test_env_token")
    values = {"base_url": "http://iris:8080", "allow_container_host": True}
    assert Config.from_mapping(values).token == ""
    configured = Config.from_mapping({**values, "token": "iris_test_token"})
    assert configured.token == "iris_test_token"
    assert "iris_test_token" not in repr(configured)
