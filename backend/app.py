import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.health_routes import router as health_router
from api.websocket_routes import router as websocket_router
from utils.logger import get_logger


logger = get_logger(__name__)


# =========================================================
# STATIC AUDIO FILES
# =========================================================

LIMIT_AUDIO_PATH = (
    Path(__file__).resolve().parent
    / "audio"
    / "static"
    / "limit-reached.mp3"
)


# =========================================================
# STARTUP SERVICES
# =========================================================

async def _init_memory() -> None:
    try:
        from memory.memory_manager import memory_manager

        await memory_manager.initialize()

        logger.info(
            "MongoDB memory ready ✅"
        )

    except Exception as exc:
        logger.error(
            f"MongoDB initialization failed: {exc}"
        )


async def _warmup_services() -> None:
    try:
        from rag.vector_store import vector_store

        await asyncio.to_thread(
            vector_store._ensure_index
        )

        logger.info(
            "Pinecone ready ✅"
        )

    except Exception as exc:
        logger.warning(
            f"Pinecone warmup failed: {exc}"
        )


# =========================================================
# APP LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "AI Voice Agent (Sada) starting..."
    )

    await asyncio.gather(
        _init_memory(),
        _warmup_services(),
    )

    logger.info(
        "AI Voice Agent ready ✅"
    )

    yield

    try:
        from memory.memory_manager import memory_manager

        await memory_manager.close()

    except Exception as exc:
        logger.warning(
            f"MongoDB shutdown error: {exc}"
        )

    logger.info(
        "AI Voice Agent shut down cleanly"
    )


# =========================================================
# CREATE APP
# =========================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Voice Agent — Sada",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # =====================================================
    # LIMIT REACHED AUDIO
    # =====================================================

    @app.get(
        "/limit-reached.mp3",
        include_in_schema=False,
    )
    async def limit_reached_audio():
        if not LIMIT_AUDIO_PATH.is_file():
            logger.error(
                f"Limit audio not found: {LIMIT_AUDIO_PATH}"
            )

            raise HTTPException(
                status_code=404,
                detail="Limit audio not found",
            )

        return FileResponse(
            path=LIMIT_AUDIO_PATH,
            media_type="audio/mpeg",
        )


    # =====================================================
    # ROUTERS
    # =====================================================

    app.include_router(
        health_router
    )

    app.include_router(
        websocket_router
    )

    return app


app = create_app()