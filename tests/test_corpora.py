from pathlib import Path

from reciprocator import (
    available_corpora,
    corpus_path,
    get_corpus,
    read_corpus_readme,
    read_corpus_sources,
    read_corpus_text,
)


def test_available_corpora_match_imported_registry() -> None:
    assert [corpus.name for corpus in available_corpora()] == [
        "chinese_classics",
        "greek_classics",
    ]


def test_corpus_path_exposes_expected_layout() -> None:
    for name in ("chinese_classics", "greek_classics"):
        corpus = get_corpus(name)
        repo_root = Path(__file__).resolve().parents[1]

        with corpus_path(corpus.name) as path:
            assert isinstance(path, Path)
            assert path.is_dir()
            assert path == repo_root / "corpora" / corpus.name
            assert (path / corpus.combined_filename).is_file()
            assert (path / corpus.readme_filename).is_file()
            assert (path / corpus.sources_filename).is_file()


def test_greek_corpus_has_provenance_rows() -> None:
    sources = read_corpus_sources("greek_classics").strip().splitlines()

    assert sources[0] == "author\tebook_id\ttitle\turl\tfilename"
    assert len(sources) > 5


def test_corpus_text_and_readme_are_nonempty() -> None:
    for name in ("chinese_classics", "greek_classics"):
        text = read_corpus_text(name)
        readme = read_corpus_readme(name)

        assert len(text) > 1000
        assert len(readme) > 10
