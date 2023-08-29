"""Fetches documents from a github repo."""
import os
from pathlib import Path

import joblib
from pydantic import BaseModel, Field


class DocumentFetcher(BaseModel):
    """Fetches documents from a github repo."""

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
            verbose=True,
            concurrent_requests=10,
        )

    async def _fetch_docs(self, loader):
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
        ]
        documents = await loader._generate_documents(
            blobs_and_paths=blobs_and_paths_to_keep
        )
        return documents

    async def run(self):
        """Run the document fetcher."""
        self.setup()
        loader = self._build_loader()
        documents = await self._fetch_docs(loader)
        return documents

    def __hash__(self) -> int:
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)
