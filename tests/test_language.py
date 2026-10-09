import pytest

from agent.language import detect_language, language_name


@pytest.mark.parametrize(
    "text,code",
    [
        ("API response time increased 300% after deploying v2.3.1", "en"),
        ("La latencia de la API aumentó un 300% después del despliegue", "es"),
        ("Bestellbestätigungen kommen mit einer Stunde Verspätung an", "de"),
        ("Le point de terminaison de paiement renvoie des erreurs 500", "fr"),
        ("レディスへの接続がタイムアウトしてAPIが遅くなりました", "ja"),
    ],
)
def test_detects_language(text, code):
    assert detect_language(text) == code


def test_short_or_empty_text_defaults_to_english():
    assert detect_language("") == "en"
    assert detect_language("500 error") == "en"


def test_language_name_falls_back_to_code():
    assert language_name("es") == "Spanish"
    assert language_name("xx") == "xx"
