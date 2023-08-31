"""Fetches documents from a github repo."""
# from functools import partial
import os
from pathlib import Path
from typing import Optional

import joblib

# from typing import TYPE_CHECKING
from pydantic import BaseModel, Field

# if TYPE_CHECKING:
#     from llama_index.schema import Document


class GithubDocumentLoader(BaseModel):
    """Loads documents from a github repo."""

    owner: str = Field(
        description="Github owner of the repository to fetch documents from."
    )

    repo: str = Field(description="Github repository to fetch documents from.")

    version_tag: str = Field(
        description="Version tag of the repository to fetch documents from."
    )

    github_token: str = Field(
        description=(
            "Github token to use for authentication. If not provided, will use the"
            "GITHUB_TOKEN environment variable.",
        ),
        default_factory=lambda: os.environ.get("GITHUB_TOKEN"),
        repr=False,
        exclude=True,
    )

    paths_to_include: list[str] = Field(
        description="List of paths to include in the document generation.",
    )

    file_extensions_to_include: list[str] = Field(
        description="List of file extensions to include in the document generation.",
    )

    filenames_to_exclude: list[str] = Field(
        description="List of file names to exclude in the document generation.",
    )

    paths_to_exclude: list[str] = Field(
        description="List of paths to exclude in the document generation.",
    )

    def setup(self):
        """Download the loader from llama hub."""
        from llama_index.readers.download import download_loader, LOADER_HUB_URL

        download_loader(
            loader_class="GithubRepositoryReader",
            loader_hub_url=LOADER_HUB_URL,
            refresh_cache=False,
            use_gpt_index_import=False,
            custom_path=None,
        )

    def _build_loader(self):
        from llama_hub.github_repo import GithubClient, GithubRepositoryReader

        github_client = GithubClient(self.github_token)
        return GithubRepositoryReader(
            github_client,
            owner=self.owner,
            repo=self.repo,
            filter_directories=(
                self.paths_to_include,
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            filter_file_extensions=(
                self.file_extensions_to_include,
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            concurrent_requests=10,
            use_parser=False,
            verbose=False,
        )

    async def _fetch_docs(self, loader, limit):
        branch_data = await loader._github_client.get_branch(
            loader._owner, loader._repo, branch=self.version_tag
        )
        tree_sha = branch_data.commit.commit.tree.sha
        blobs_and_paths = await loader._recurse_tree(tree_sha)
        blobs_and_paths_to_keep = [
            blob_and_path
            for blob_and_path in blobs_and_paths
            if all(
                path_to_exclude not in blob_and_path[1]  # blob_and_path[1] is the path
                for path_to_exclude in self.paths_to_exclude
            )
            and all(
                file_name_to_exclude
                not in Path(blob_and_path[1]).name  # blob_and_path[1] is the path
                for file_name_to_exclude in self.filenames_to_exclude
            )
        ][:limit]
        documents = await loader._generate_documents(
            blobs_and_paths=blobs_and_paths_to_keep
        )
        return documents

    async def run(self, limit: Optional[int] = None):
        """Run the document fetcher."""
        self.setup()
        loader = self._build_loader()
        documents = await self._fetch_docs(loader, limit=limit)
        return documents

    def __hash__(self) -> int:
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)


# class DirectoryDocumentFetcher(BaseModel):
#     """Fetches documents from a directory."""

#     directory: str = Field(
#         description="Directory to fetch documents from.",
#     )

#     paths_to_include: list[str] = Field(
#         description="List of paths to include in the document generation.",
#     )

#     file_extensions_to_include: list[str] = Field(
#         description="List of file extensions to include in the document generation.",
#     )

#     filenames_to_exclude: list[str] = Field(
#         description="List of file names to exclude in the document generation.",
#     )

#     paths_to_exclude: list[str] = Field(
#         description="List of paths to exclude in the document generation.",
#     )

#     def _build_loader(self):
#         from llama_index import download_loader

#         UnstructuredReader = download_loader("UnstructuredReader")
#         return UnstructuredReader()

#     def run(self):
#         """Run the document fetcher."""
#         import ray

#         ray.init(ignore_reinit_error=True)

#         # Get the paths for the locally downloaded documentation.
#         all_docs_gen = Path(self.directory).rglob("*")
#         all_docs = [{"path": doc.resolve()} for doc in all_docs_gen][:100]

#         # Create the Ray Dataset pipeline
#         ds = ray.data.from_items(all_docs)

#         # Use `flat_map` since there is a 1:N relationship.
#         # Each filepath returns multiple documents.
#         loaded_docs = ds.flat_map(
#             partial(self._load_and_parse_files, loader=self._build_loader())
#         )

#         return [doc for doc in loaded_docs.iter_rows()]

#     def __hash__(self) -> int:
#         hash_hex = joblib.hash(self.dict())
#         return int(hash_hex, 16)

#     def _load_and_parse_files(
#         self, file_row: dict[str, Path], *, loader
#     ) -> list[dict[str, "Document"]]:
#         """Load and parse files."""
#         documents = []
#         file = file_row["path"]
#         if file.suffix.lower() == ".html":
#             loaded_doc = loader.load_data(file=file, split_documents=False)
#             print("loaded_doc", loaded_doc)
#             loaded_doc.metadata["path"] = str(file)
#             documents.extend(loaded_doc)
#         return documents
