
import pandas as pd
import psycopg2

def get_mimic_notes(dbname, user, password, host, port=5432, n=1000):
    conn = psycopg2.connect(
        dbname=dbname, user=user, password=password, host=host, port=port
    )
    sql = """
        SELECT subject_id, hadm_id, charttime, category, text
        FROM noteevents
        WHERE (category IN ('Discharge summary', 'Radiology', 'Echo'))
        AND (
            text ILIKE '%pulmonary embolism%' OR
            text ILIKE '%anticoag%' OR
            text ILIKE '%heparin%' OR
            text ILIKE '%warfarin%' OR
            text ILIKE '%apixaban%'
        )
        LIMIT %s;
    """
    df = pd.read_sql(sql, conn, params=(n,))
    df.to_csv('data/mimic_extracted.csv', index=False)
    print(f"Wrote {len(df)} notes to data/mimic_extracted.csv")
    return df

if __name__ == "__main__":
    # Edit these before running!
    get_mimic_notes(dbname='mimic', user='postgres', password='password', host='localhost')
