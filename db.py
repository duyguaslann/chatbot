import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
import os

load_dotenv()
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)

conn.set_client_encoding('UTF8')

#user üreten fonks
def generate_next_user_id():
    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_id FROM chats
            WHERE user_id ~ '^user\\d+$'
            ORDER BY LENGTH(user_id) DESC, user_id DESC
            LIMIT 1;
        """)
        result = cur.fetchone()

        if result:
            last_user_id = result[0]
            last_number = int(last_user_id.replace("user", ""))
            new_user_id = f"user{last_number + 1}"
        else:
            new_user_id = "user1"

        return new_user_id


def get_user_fullname(user_id):
    with conn.cursor() as cur:
        cur.execute("SELECT first_name, last_name FROM users WHERE id = %s", (user_id,))
        result = cur.fetchone()
        if result:
            return f"{result[0]} {result[1]}"
        return None

#calling
def get_user_profile(user_id):
    with conn.cursor() as cur:
        cur.execute("SELECT first_name, last_name, age FROM users WHERE id = %s", (user_id,))
        result = cur.fetchone()
    if not result:
        return "Kullanıcı bilgileri bulunamadı."
    firstname, lastname, age = result
    return f"Adın: {firstname} Soyadın: {lastname}, {age} yaşındasın."

def create_chat(user_id, title="Yeni Sohbet"):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING *",
            (user_id, title)
        )
        chat = cur.fetchone()
        conn.commit()

    fullname = get_user_fullname(user_id)
    welcome_msg = f"Hello Duygu Aslan, what would you like to learn?" if fullname else "Merhaba, ne öğrenmek istiyorsun?"

    save_message(user_type=0, message_text=welcome_msg, chat_id=chat["id"], status=1)
    return chat

def save_message(user_type, message_text, chat_id, status=1):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (user_type, message_text, chat_id, status) VALUES (%s, %s, %s, %s)",
            (user_type, message_text, chat_id, status)
        )
        conn.commit()

def get_user_by_email(email):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, first_name, last_name, email, password FROM users WHERE email = %s", (email,))
        return cur.fetchone()

#daily messages limit
def get_today_message_count(user_id):
    with conn.cursor() as cursor:
        query = """
            SELECT COUNT(*) FROM messages
            WHERE user_type = 1
            AND DATE(created_at) = CURRENT_DATE
            AND chat_id IN (
                SELECT id FROM chats WHERE user_id = %s
            )
        """
        cursor.execute(query, (user_id,))
        count = cursor.fetchone()[0]
        return count


#mesajları getir
def get_messages_by_chat(chat_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_type, message_text, created_at FROM messages WHERE chat_id = %s AND status = 1 ORDER BY created_at",
            (chat_id,)
        )
        rows = cur.fetchall()
        return [{"user_type": row[0], "message_text": row[1], "created_at": row[2].isoformat()} for row in rows]

#history limit
def get_last_messages(chat_id, limit=10):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_type, message_text, created_at
            FROM messages
            WHERE chat_id = %s AND status = 1
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (chat_id, limit)
        )
        rows = cur.fetchall()
        rows.reverse() #en eski tarih en başta
        return [{"user_type": row[0], "message_text": row[1], "created_at": row[2].isoformat()} for row in rows]


def get_chats(user_id):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM chats WHERE status = 1 AND user_id = %s ORDER BY created_at DESC", (user_id,))
        return cur.fetchall()


def update_chat_title(chat_id, new_title):
    with conn.cursor() as cur:
        cur.execute("UPDATE chats SET title = %s WHERE id = %s", (new_title, chat_id))
        conn.commit()

#status=0
def delete_chat(chat_id):
    with conn.cursor() as cur:
        # Mesajları soft delete yap
        cur.execute("UPDATE messages SET status = 0 WHERE chat_id = %s", (chat_id,))
        # Chat’i soft delete yap
        cur.execute("UPDATE chats SET status = 0 WHERE id = %s", (chat_id,))
        conn.commit()

def clear_chat_history(chat_id):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE messages SET status = 0 WHERE chat_id = %s",
            (chat_id,)
        )
        conn.commit()

