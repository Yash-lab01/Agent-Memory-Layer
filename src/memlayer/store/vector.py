"""Qdrant vector storage adapter implementing VectorStoreBase."""

import warnings
from typing import Any, cast
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from memlayer.store.vector_base import VectorPoint, VectorSearchResult, VectorStoreBase


class QdrantVectorStore(VectorStoreBase):
    """Asynchronous Qdrant vector store adapter supporting in-memory, disk, and server modes."""

    def __init__(
        self,
        location: str = ":memory:",
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "memlayer_memories",
        vector_size: int = 384,
        distance: models.Distance = models.Distance.COSINE,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        if client is not None:
            self.client = client
        elif url is not None:
            self.client = AsyncQdrantClient(url=url, api_key=api_key)
        else:
            self.client = AsyncQdrantClient(location=location)

    async def init_store(self) -> None:
        """Create collection and payload indexes if not already existing."""
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="user_id",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="status",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

    async def upsert(self, user_id: str, points: list[VectorPoint]) -> None:
        """Insert or update vector points, enforcing user_id in payload."""
        if not points:
            return

        clean_user_id = user_id.strip()
        qdrant_points: list[models.PointStruct] = []

        for p in points:
            payload: dict[str, Any] = dict(p.payload)
            payload["user_id"] = clean_user_id
            qdrant_points.append(
                models.PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=payload,
                )
            )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points,
        )

    async def search(
        self,
        user_id: str,
        query_vector: list[float],
        limit: int = 10,
        status: str | None = "active",
    ) -> list[VectorSearchResult]:
        """Perform similarity search strictly constrained to user_id."""
        clean_user_id = user_id.strip()

        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=clean_user_id),
            )
        ]
        if status is not None:
            conditions.append(
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value=status),
                )
            )

        query_filter = models.Filter(must=conditions)

        res = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

        return [
            VectorSearchResult(
                id=str(hit.id),
                score=float(hit.score),
                payload=dict(hit.payload or {}),
            )
            for hit in res.points
        ]

    async def delete(self, user_id: str, point_ids: list[str]) -> None:
        """Delete specific vector points belonging strictly to user_id."""
        if not point_ids:
            return

        clean_user_id = user_id.strip()
        has_id_list = cast(list[int | str | UUID], point_ids)
        filter_selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=clean_user_id),
                    ),
                    models.HasIdCondition(has_id=has_id_list),
                ]
            )
        )
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_selector,
        )

    async def delete_user_vectors(self, user_id: str) -> int:
        """Purge all vector embeddings belonging to user_id."""
        clean_user_id = user_id.strip()
        filter_selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=clean_user_id),
                    )
                ]
            )
        )
        count_res = await self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_selector.filter,
        )
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_selector,
        )
        return int(count_res.count)

    async def close(self) -> None:
        """Close connection to Qdrant."""
        await self.client.close()
