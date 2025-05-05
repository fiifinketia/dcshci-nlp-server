"""
FastAPI application for the TTS server.
"""
import os
import logging
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
from routes import router
from config import (
    API_PREFIX, 
    API_TITLE, 
    API_DESCRIPTION, 
    API_VERSION, 
    CORS_ORIGINS,
    DEBUG
)

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
    debug=DEBUG
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import API routes

# Include API router
app.include_router(router, prefix=API_PREFIX)

# Add favicon route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return the favicon."""
    return FileResponse("static/favicon.ico") if os.path.exists("static/favicon.ico") else None

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
        },
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logging.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if DEBUG else "Internal server error"
        }
    )