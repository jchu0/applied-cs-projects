"""Document parsers for various file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from .schemas import Document, generate_id


class DocumentParser(ABC):
    """Base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Parse file into document."""
        pass

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Check if parser supports file type."""
        pass


class PDFParser(DocumentParser):
    """Parse PDF documents using PyMuPDF."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> Document:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required. Install with: pip install PyMuPDF")

        doc = fitz.open(file_path)
        text_parts = []

        for page in doc:
            text = page.get_text()
            text_parts.append(text)

        content = "\n\n".join(text_parts)

        return Document(
            id=generate_id(str(file_path)),
            content=content,
            metadata={
                "filename": file_path.name,
                "num_pages": len(doc),
                "file_type": "pdf",
            },
            source=str(file_path),
        )


class HTMLParser(DocumentParser):
    """Parse HTML documents using BeautifulSoup."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".html", ".htm"]

    def parse(self, file_path: Path) -> Document:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("BeautifulSoup required. Install with: pip install beautifulsoup4")

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        content = "\n".join(line for line in lines if line)

        # Extract title
        title = soup.title.string if soup.title else file_path.stem

        return Document(
            id=generate_id(str(file_path)),
            content=content,
            metadata={
                "filename": file_path.name,
                "title": title,
                "file_type": "html",
            },
            source=str(file_path),
        )


class MarkdownParser(DocumentParser):
    """Parse Markdown documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".md", ".markdown"]

    def parse(self, file_path: Path) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Try to convert via markdown, fall back to plain text
        try:
            import markdown
            from bs4 import BeautifulSoup

            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, "html.parser")
            content = soup.get_text(separator="\n")
        except ImportError:
            # Fallback: just use raw markdown
            content = md_content

        # Extract title from first heading
        title = file_path.stem
        lines = md_content.split("\n")
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return Document(
            id=generate_id(str(file_path)),
            content=content,
            metadata={
                "filename": file_path.name,
                "title": title,
                "file_type": "markdown",
            },
            source=str(file_path),
        )


class TextParser(DocumentParser):
    """Parse plain text documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".txt", ".text"]

    def parse(self, file_path: Path) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return Document(
            id=generate_id(str(file_path)),
            content=content,
            metadata={
                "filename": file_path.name,
                "file_type": "text",
            },
            source=str(file_path),
        )


class DocumentIngestion:
    """Main ingestion pipeline."""

    def __init__(self):
        self.parsers = [
            PDFParser(),
            HTMLParser(),
            MarkdownParser(),
            TextParser(),
        ]

    def add_parser(self, parser: DocumentParser):
        """Add custom parser."""
        self.parsers.insert(0, parser)  # Custom parsers take priority

    def ingest(self, file_path: Path) -> Document:
        """Ingest single file."""
        file_path = Path(file_path)

        for parser in self.parsers:
            if parser.supports(file_path):
                return parser.parse(file_path)

        raise ValueError(f"No parser found for {file_path}")

    def ingest_directory(self, dir_path: Path) -> Iterator[Document]:
        """Ingest all supported files in directory."""
        dir_path = Path(dir_path)

        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    yield self.ingest(file_path)
                except ValueError:
                    continue  # Skip unsupported files

    def supported_extensions(self) -> list[str]:
        """List all supported file extensions."""
        extensions = []
        for parser in self.parsers:
            if isinstance(parser, PDFParser):
                extensions.append(".pdf")
            elif isinstance(parser, HTMLParser):
                extensions.extend([".html", ".htm"])
            elif isinstance(parser, MarkdownParser):
                extensions.extend([".md", ".markdown"])
            elif isinstance(parser, TextParser):
                extensions.extend([".txt", ".text"])
        return extensions
