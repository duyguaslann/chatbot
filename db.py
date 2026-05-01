import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
import os
import bcrypt

load_dotenv()
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost")),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)

conn.set_client_encoding('UTF8')


def generate_next_user_id():
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_id FROM chats
                WHERE user_id ~ '^user\\d+$'
                ORDER BY LENGTH(user_id) DESC, user_id DESC
                LIMIT 1;
            """)
            result = cur.fetchone()
        if result:
            last_number = int(result[0].replace("user", ""))
            return f"user{last_number + 1}"
        return "user1"
    except Exception:
        conn.rollback()
        raise


def get_user_fullname(user_id):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT first_name, last_name FROM users WHERE id = %s", (user_id,))
            result = cur.fetchone()
        if result:
            return f"{result[0]} {result[1]}"
        return None
    except Exception:
        conn.rollback()
        raise


def get_user_profile(user_id):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT first_name, last_name, age FROM users WHERE id = %s", (user_id,))
            result = cur.fetchone()
        if not result:
            return "Kullanıcı bilgileri bulunamadı."
        firstname, lastname, age = result
        return f"Adın: {firstname} Soyadın: {lastname}, {age} yaşındasın."
    except Exception:
        conn.rollback()
        raise


def create_chat(user_id, title="Yeni Sohbet"):
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING *",
                (user_id, title)
            )
            chat = cur.fetchone()
            conn.commit()
    except Exception:
        conn.rollback()
        raise

    fullname = get_user_fullname(user_id)
    welcome_msg = (
        f"Hello {fullname}, what would you like to learn?"
        if fullname else "Merhaba, ne öğrenmek istiyorsun?"
    )
    save_message(user_type=0, message_text=welcome_msg, chat_id=chat["id"], status=1)
    return chat


def save_message(user_type, message_text, chat_id, status=1):
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (user_type, message_text, chat_id, status) VALUES (%s, %s, %s, %s)",
                (user_type, message_text, chat_id, status)
            )
            conn.commit()
    except Exception:
        conn.rollback()
        raise


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def get_user_by_email(email):
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, first_name, last_name, email, password FROM users WHERE email = %s",
                (email,)
            )
            return cur.fetchone()
    except Exception:
        conn.rollback()
        raise


def get_today_message_count(user_id):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM messages
                WHERE user_type = 1
                AND DATE(created_at) = CURRENT_DATE
                AND chat_id IN (SELECT id FROM chats WHERE user_id = %s)
                """,
                (user_id,)
            )
            return cur.fetchone()[0]
    except Exception:
        conn.rollback()
        raise


def get_messages_by_chat(chat_id):
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_type, message_text, created_at FROM messages "
                "WHERE chat_id = %s AND status = '1' ORDER BY created_at",
                (chat_id,)
            )
            rows = cur.fetchall()
        return [
            {"user_type": r[0], "message_text": r[1], "created_at": r[2].isoformat()}
            for r in rows
        ]
    except Exception:
        conn.rollback()
        raise


def get_last_messages(chat_id, limit=10):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_type, message_text, created_at
                FROM messages
                WHERE chat_id = %s AND status = '1'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (chat_id, limit)
            )
            rows = cur.fetchall()
        rows.reverse()
        return [
            {"user_type": r[0], "message_text": r[1], "created_at": r[2].isoformat()}
            for r in rows
        ]
    except Exception:
        conn.rollback()
        raise


def get_chats(user_id):
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM chats WHERE status = '1' AND user_id = %s ORDER BY created_at DESC",
                (user_id,)
            )
            return cur.fetchall()
    except Exception:
        conn.rollback()
        raise


def update_chat_title(chat_id, new_title):
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE chats SET title = %s WHERE id = %s", (new_title, chat_id))
            conn.commit()
    except Exception:
        conn.rollback()
        raise


def delete_chat(chat_id):
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE messages SET status = 0 WHERE chat_id = %s", (chat_id,))
            cur.execute("UPDATE chats SET status = 0 WHERE id = %s", (chat_id,))
            conn.commit()
    except Exception:
        conn.rollback()
        raise


def clear_chat_history(chat_id):
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE messages SET status = 0 WHERE chat_id = %s", (chat_id,))
            conn.commit()
    except Exception:
        conn.rollback()
        raise
