"""Tests for text transformers."""

import numpy as np
import pytest

from feature_platform.transformers.text import (
    TextCleaner,
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
    NGramExtractor,
    TextStatistics,
)


class TestTextCleaner:
    """Tests for TextCleaner."""

    def test_lowercase(self):
        """Test lowercase conversion."""
        X = np.array([["Hello World"]])
        cleaner = TextCleaner(lowercase=True)
        result = cleaner.fit_transform(X)

        assert result[0, 0] == "hello world"

    def test_remove_punctuation(self):
        """Test punctuation removal."""
        X = np.array([["Hello, World!"]])
        cleaner = TextCleaner(remove_punctuation=True)
        result = cleaner.fit_transform(X)

        assert "," not in result[0, 0]
        assert "!" not in result[0, 0]

    def test_remove_numbers(self):
        """Test number removal."""
        X = np.array([["abc 123 def"]])
        cleaner = TextCleaner(remove_numbers=True)
        result = cleaner.fit_transform(X)

        assert "123" not in result[0, 0]

    def test_remove_stopwords(self):
        """Test stopword removal."""
        X = np.array([["the quick brown fox"]])
        cleaner = TextCleaner(remove_stopwords=True)
        result = cleaner.fit_transform(X)

        assert "the" not in result[0, 0].split()

    def test_min_length(self):
        """Test minimum token length."""
        X = np.array([["a ab abc abcd"]])
        cleaner = TextCleaner(min_length=3)
        result = cleaner.fit_transform(X)

        tokens = result[0, 0].split()
        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        X = np.array([["hello    world  test"]])
        cleaner = TextCleaner(remove_whitespace=True)
        result = cleaner.fit_transform(X)

        assert "  " not in result[0, 0]

    def test_custom_stopwords(self):
        """Test custom stopwords."""
        X = np.array([["foo bar baz"]])
        cleaner = TextCleaner(
            remove_stopwords=True,
            custom_stopwords={"foo", "bar"}
        )
        result = cleaner.fit_transform(X)

        tokens = result[0, 0].split()
        assert "foo" not in tokens
        assert "bar" not in tokens
        assert "baz" in tokens


class TestCountVectorizer:
    """Tests for CountVectorizer."""

    def test_basic_vectorization(self):
        """Test basic count vectorization."""
        X = np.array([
            ["the cat sat"],
            ["the dog ran"],
        ])
        vectorizer = CountVectorizer()
        result = vectorizer.fit_transform(X)

        # Should have vocabulary from both docs
        assert result.shape[1] > 0
        assert vectorizer.is_fitted

    def test_max_features(self):
        """Test max features limit."""
        X = np.array([
            ["a b c d e f g h i j"],
        ])
        vectorizer = CountVectorizer(max_features=5)
        result = vectorizer.fit_transform(X)

        assert result.shape[1] == 5

    def test_binary_mode(self):
        """Test binary count mode."""
        X = np.array([["word word word"]])
        vectorizer = CountVectorizer(binary=True)
        result = vectorizer.fit_transform(X)

        # In binary mode, count should be 1 not 3
        assert result[0, 0] == 1

    def test_ngram_range(self):
        """Test n-gram extraction."""
        X = np.array([["a b c"]])
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        result = vectorizer.fit_transform(X)

        # Should include unigrams and bigrams
        assert "a b" in vectorizer.vocabulary_ or "b c" in vectorizer.vocabulary_

    def test_min_df(self):
        """Test minimum document frequency."""
        X = np.array([
            ["common word"],
            ["common term"],
            ["common phrase"],
        ])
        vectorizer = CountVectorizer(min_df=2)
        vectorizer.fit(X)

        # Only "common" appears in all docs
        assert "common" in vectorizer.vocabulary_

    def test_max_df(self):
        """Test maximum document frequency."""
        X = np.array([
            ["the cat"],
            ["the dog"],
            ["the bird"],
        ])
        vectorizer = CountVectorizer(max_df=0.5)
        vectorizer.fit(X)

        # "the" appears in all docs (100%), should be filtered
        assert "the" not in vectorizer.vocabulary_


class TestTfidfVectorizer:
    """Tests for TfidfVectorizer."""

    def test_basic_tfidf(self):
        """Test basic TF-IDF vectorization."""
        X = np.array([
            ["the cat sat"],
            ["the dog ran"],
        ])
        vectorizer = TfidfVectorizer()
        result = vectorizer.fit_transform(X)

        assert result.shape[0] == 2
        assert vectorizer.is_fitted

    def test_l2_normalization(self):
        """Test L2 normalization."""
        X = np.array([["word1 word2 word3"]])
        vectorizer = TfidfVectorizer(norm="l2")
        result = vectorizer.fit_transform(X)

        # L2 norm should be 1
        l2_norm = np.sqrt((result ** 2).sum())
        assert l2_norm == pytest.approx(1.0)

    def test_l1_normalization(self):
        """Test L1 normalization."""
        X = np.array([["word1 word2 word3"]])
        vectorizer = TfidfVectorizer(norm="l1")
        result = vectorizer.fit_transform(X)

        # L1 norm should be 1
        l1_norm = np.abs(result).sum()
        assert l1_norm == pytest.approx(1.0)

    def test_no_idf(self):
        """Test without IDF."""
        X = np.array([
            ["word word word"],
            ["word other"],
        ])
        vectorizer = TfidfVectorizer(use_idf=False, norm=None)
        result = vectorizer.fit_transform(X)

        # Without IDF, should just be term frequency
        assert vectorizer.is_fitted

    def test_sublinear_tf(self):
        """Test sublinear TF scaling."""
        X = np.array([["word word word word"]])
        vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=False, norm=None)
        result = vectorizer.fit_transform(X)

        # With sublinear TF, log(4) + 1 instead of 4
        # value should be less than 4
        assert result[0, 0] < 4


class TestHashingVectorizer:
    """Tests for HashingVectorizer."""

    def test_fixed_dimensions(self):
        """Test fixed output dimensions."""
        X = np.array([
            ["many different words here"],
            ["even more unique vocabulary"],
        ])
        vectorizer = HashingVectorizer(n_features=16)
        result = vectorizer.fit_transform(X)

        assert result.shape == (2, 16)

    def test_handles_unseen_words(self):
        """Test handling of unseen words."""
        X_train = np.array([["hello world"]])
        X_test = np.array([["completely new text"]])

        vectorizer = HashingVectorizer(n_features=32)
        vectorizer.fit(X_train)
        result = vectorizer.transform(X_test)

        assert result.shape == (1, 32)

    def test_normalization(self):
        """Test L2 normalization."""
        X = np.array([["test text here"]])
        vectorizer = HashingVectorizer(n_features=16, norm="l2")
        result = vectorizer.fit_transform(X)

        l2_norm = np.sqrt((result ** 2).sum())
        assert l2_norm == pytest.approx(1.0)

    def test_alternate_sign(self):
        """Test alternate sign feature."""
        X = np.array([["word word word"]])
        vectorizer = HashingVectorizer(n_features=32, alternate_sign=True)
        result = vectorizer.fit_transform(X)

        # With alternate sign, some values might be negative
        # The sum might not equal the word count


class TestNGramExtractor:
    """Tests for NGramExtractor."""

    def test_character_ngrams(self):
        """Test character n-gram extraction."""
        X = np.array([["hello"]])
        extractor = NGramExtractor(n=2, level="char")
        result = extractor.fit_transform(X)

        # "hello" has 4 character bigrams: he, el, ll, lo
        assert result.shape[1] > 0

    def test_word_ngrams(self):
        """Test word n-gram extraction."""
        X = np.array([["one two three four"]])
        extractor = NGramExtractor(n=2, level="word")
        result = extractor.fit_transform(X)

        # Should have word bigrams
        assert "one two" in extractor.vocabulary_ or "two three" in extractor.vocabulary_

    def test_max_features(self):
        """Test max features limit."""
        X = np.array([["a b c d e f g h i j"]])
        extractor = NGramExtractor(n=1, level="word", max_features=5)
        result = extractor.fit_transform(X)

        assert result.shape[1] == 5


class TestTextStatistics:
    """Tests for TextStatistics."""

    def test_length(self):
        """Test text length calculation."""
        X = np.array([["hello"]])
        stats = TextStatistics(features=["length"])
        result = stats.fit_transform(X)

        assert result[0, 0] == 5

    def test_word_count(self):
        """Test word count."""
        X = np.array([["one two three"]])
        stats = TextStatistics(features=["word_count"])
        result = stats.fit_transform(X)

        assert result[0, 0] == 3

    def test_avg_word_length(self):
        """Test average word length."""
        X = np.array([["a bb ccc"]])  # lengths: 1, 2, 3
        stats = TextStatistics(features=["avg_word_length"])
        result = stats.fit_transform(X)

        assert result[0, 0] == pytest.approx(2.0)

    def test_digit_count(self):
        """Test digit count."""
        X = np.array([["abc 123 def 456"]])
        stats = TextStatistics(features=["digit_count"])
        result = stats.fit_transform(X)

        assert result[0, 0] == 6

    def test_upper_count(self):
        """Test uppercase count."""
        X = np.array([["Hello WORLD"]])
        stats = TextStatistics(features=["upper_count"])
        result = stats.fit_transform(X)

        assert result[0, 0] == 6  # H, W, O, R, L, D

    def test_punct_count(self):
        """Test punctuation count."""
        X = np.array([["Hello, World! How are you?"]])
        stats = TextStatistics(features=["punct_count"])
        result = stats.fit_transform(X)

        assert result[0, 0] == 3  # , ! ?

    def test_unique_words(self):
        """Test unique word count."""
        X = np.array([["the the quick quick fox"]])
        stats = TextStatistics(features=["unique_words"])
        result = stats.fit_transform(X)

        assert result[0, 0] == 3  # the, quick, fox

    def test_multiple_features(self):
        """Test multiple features."""
        X = np.array([["Hello World"]])
        stats = TextStatistics(features=["length", "word_count", "unique_words"])
        result = stats.fit_transform(X)

        assert result.shape == (1, 3)


class TestTextTransformersIntegration:
    """Integration tests for text transformers."""

    def test_clean_then_vectorize(self):
        """Test cleaning then vectorizing text."""
        X = np.array([
            ["Hello, World! This is a TEST."],
            ["Another TEST document here."],
        ])

        # Clean
        cleaner = TextCleaner(lowercase=True, remove_punctuation=True)
        cleaned = cleaner.fit_transform(X)

        # Vectorize
        vectorizer = TfidfVectorizer(max_features=10)
        result = vectorizer.fit_transform(cleaned)

        assert result.shape[0] == 2

    def test_multiple_text_columns(self):
        """Test with multiple text columns."""
        X = np.array([
            ["hello", "world"],
            ["foo", "bar"],
        ])

        vectorizer = CountVectorizer(max_features=10)
        result = vectorizer.fit_transform(X)

        assert result.shape[0] == 2


class TestTextEdgeCases:
    """Test edge cases for text transformers."""

    def test_empty_string(self):
        """Test handling of empty strings."""
        X = np.array([[""]])
        cleaner = TextCleaner()
        result = cleaner.fit_transform(X)

        assert result[0, 0] == ""

    def test_single_word(self):
        """Test single word text."""
        X = np.array([["word"]])
        vectorizer = CountVectorizer()
        result = vectorizer.fit_transform(X)

        assert result.shape[1] == 1

    def test_unicode_text(self):
        """Test unicode text handling."""
        X = np.array([["cafe nihao"]])
        vectorizer = CountVectorizer()
        result = vectorizer.fit_transform(X)

        assert vectorizer.is_fitted

    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = " ".join(["word"] * 1000)
        X = np.array([[long_text]])

        vectorizer = HashingVectorizer(n_features=64)
        result = vectorizer.fit_transform(X)

        assert result.shape == (1, 64)

    def test_special_characters(self):
        """Test handling of special characters."""
        X = np.array([["<html>test</html> @user #hashtag"]])
        cleaner = TextCleaner(remove_punctuation=True)
        result = cleaner.fit_transform(X)

        # Should remove special characters
        assert "<" not in result[0, 0]
