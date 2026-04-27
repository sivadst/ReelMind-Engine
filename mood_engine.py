MOODS = {
    "Happy":       {"emoji":"😄","desc":"Light, fun, feel-good films","genres":["Comedy","Animation","Family"],"min_rating":6.5,"keywords":["fun","comedy","laugh","joy","adventure"]},
    "Sad":         {"emoji":"😭","desc":"Emotional, touching stories","genres":["Drama","Romance"],"min_rating":7.0,"keywords":["loss","love","emotion","grief","beautiful"]},
    "Dark":        {"emoji":"💀","desc":"Intense, dark, psychological","genres":["Thriller","Horror","Crime"],"min_rating":6.5,"keywords":["dark","crime","murder","thriller","mystery"]},
    "Mind-Blown":  {"emoji":"🧠","desc":"Thought-provoking, complex","genres":["Science Fiction","Mystery"],"min_rating":7.5,"keywords":["mind","reality","future","space","time","twist"]},
    "Excited":     {"emoji":"⚡","desc":"Action, adrenaline, epic","genres":["Action","Adventure"],"min_rating":6.5,"keywords":["action","hero","battle","war","fight","epic"]},
    "Romantic":    {"emoji":"💕","desc":"Love stories, warmth","genres":["Romance","Drama"],"min_rating":6.5,"keywords":["love","romance","relationship","heart","couple"]},
}

def mood_filter(df, mood_data, min_rating=6.0, n=9):
    import pandas as pd
    filtered = df[df['vote_average'] >= max(min_rating, mood_data['min_rating'])].copy()
    genres   = mood_data['genres']
    keywords = mood_data['keywords']

    def mood_score(row):
        score = 0
        g = str(row['genres_clean']).lower()
        k = str(row['keywords_clean']).lower()
        o = str(row['overview']).lower()
        for genre in genres:
            if genre.lower() in g: score += 3
        for kw in keywords:
            if kw in k or kw in o: score += 1
        return score

    filtered['mood_score'] = filtered.apply(mood_score, axis=1)
    filtered['year'] = filtered['year'] if 'year' in filtered.columns else 0
    return filtered[filtered['mood_score'] > 0].nlargest(n, 'mood_score')