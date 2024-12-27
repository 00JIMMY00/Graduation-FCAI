from supabase import create_client

def insert_vector(data):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table("documents").insert(data).execute()
    return response
