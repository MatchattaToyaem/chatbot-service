"""
gRPC test client for chatbot-service (server.py on localhost:50051).

Usage:
    # Run all scenarios
    python client.py

    # Run a single scenario
    python client.py standalone
    python client.py followup
    python client.py newtopic
    python client.py seed        <- inserts a test session row into PostgreSQL
"""

import sys
import grpc
import chatbot_service_pb2
import chatbot_service_pb2_grpc

SERVER = "localhost:50051"

# ── Test session UUID — must exist in chat_sessions table for context tests ──
# Run `python client.py seed` first to insert this row.
TEST_SESSION_ID = "00000000-0000-0000-0000-000000000099"


def _stub():
    channel = grpc.insecure_channel(SERVER)
    return chatbot_service_pb2_grpc.HuggingFaceServiceStub(channel), channel


def ask(prompt: str, session_id: str = "") -> None:
    stub, channel = _stub()
    with channel:
        print(f"\n{'='*60}")
        print(f"PROMPT     : {prompt}")
        print(f"SESSION_ID : {session_id or '(none)'}")
        print("─" * 60)
        response = stub.GenerateResponse(
            chatbot_service_pb2.PromptRequest(
                prompt=prompt,
                session_id=session_id,
            )
        )
        print(f"ANSWER     :\n{response.result}")
        print(f"\nCONFIDENCE : {response.confidence:.3f}")
        print(f"MODEL      : {response.model}")
        if response.sources:
            print("SOURCES    :")
            for s in response.sources:
                print(f"  - {s.file} [{s.subfolder}] score={s.score:.3f}")
        print("=" * 60)


def scenario_standalone():
    """Standard question about alcohol policy — no session context."""
    ask(
        prompt="What is the company policy on alcohol consumption at work?",
        session_id="",
    )


def scenario_followup():
    """
    Follow-up question referencing the prior alcohol policy answer.
    Requires seeded test data (run: python client.py seed).
    The refiner should detect is_context_relevant=true and expand
    'those disciplinary actions' into a self-contained query.
    """
    ask(
        prompt="What are those disciplinary actions in more detail?",
        session_id=TEST_SESSION_ID,
    )


def scenario_newtopic():
    """
    New topic — document form question sent with a session_id.
    The refiner should detect is_context_relevant=false (completely
    different topic from alcohol policy) and answer without history.
    Form code detection will boost chunks from the exact document.
    """
    ask(
        prompt="What is form OF140 and how do I fill it out?",
        session_id=TEST_SESSION_ID,
    )


def seed_test_session():
    """
    Insert a test row into chat_sessions so the context-aware scenarios work.
    Requires PostgreSQL running with the settings from .env.local.
    """
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv()

    try:
        import psycopg2
    except ImportError:
        print("psycopg2-binary not installed. Run: pip install psycopg2-binary")
        return

    history = [
        {
            "answerId": "aaaaaaaa-0000-0000-0000-000000000001",
            "question": "What is the company policy on alcohol consumption at work?",
            "answer": (
                "O'Connors has a strict zero-tolerance policy on alcohol consumption "
                "at the workplace. Employees must not consume alcohol during working "
                "hours, on company premises, or while operating company vehicles. "
                "Employees who report to work under the influence of alcohol may be "
                "subject to disciplinary actions, including termination."
            ),
            "response_date": "2026-05-09T10:00:00+00:00",
        },
        {
            "answerId": "aaaaaaaa-0000-0000-0000-000000000002",
            "question": "Can I drink alcohol at a work-sponsored event?",
            "answer": (
                "Alcohol may only be consumed at work-sponsored events where it has "
                "been explicitly approved by management. Employees are expected to "
                "drink responsibly and must not drive a company or personal vehicle "
                "if their blood alcohol level exceeds the legal limit."
            ),
            "response_date": "2026-05-09T10:03:00+00:00",
        },
    ]

    conn_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", "chatdb"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "localpass"),
        "sslmode": os.getenv("POSTGRES_SSL", "prefer"),
    }

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (id, chat_name, user_name, chat_history, created_by, status)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET chat_history = EXCLUDED.chat_history
                    """,
                    (
                        TEST_SESSION_ID,
                        "Test Session",
                        "test_user",
                        json.dumps(history),
                        "test_user",
                        "active",
                    ),
                )
            conn.commit()
        print(f"Seeded test session: {TEST_SESSION_ID}")
        print("Chat history contains 2 Q&A entries about refrigerant PPE and storage.")
    except Exception as e:
        print(f"Seed failed: {e}")
        print("Make sure PostgreSQL is running and the chat_sessions table exists.")


SCENARIOS = {
    "standalone": scenario_standalone,
    "followup": scenario_followup,
    "newtopic": scenario_newtopic,
    "seed": seed_test_session,
}

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        print("\n>>> Scenario 1: Standalone (no context)")
        scenario_standalone()

        print("\n>>> Scenario 2: Follow-up (with session context)")
        print("    NOTE: Run 'python client.py seed' first if this is your first run.")
        scenario_followup()

        print("\n>>> Scenario 3: New topic (session_id present but unrelated question)")
        scenario_newtopic()

    elif target in SCENARIOS:
        SCENARIOS[target]()
    else:
        print(f"Unknown scenario '{target}'. Choose from: {', '.join(SCENARIOS)} or 'all'")
