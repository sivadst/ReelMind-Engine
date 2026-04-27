import json, os
from datetime import datetime
from collections import Counter

PROFILE_FILE = "user_profile.json"

class UserProfile:
    def __init__(self):
        self.history  = []
        self.liked    = []
        self.disliked = []
        self.genre_counts = Counter()
        self._load()

    def _load(self):
        if os.path.exists(PROFILE_FILE):
            with open(PROFILE_FILE) as f:
                data = json.load(f)
                self.history      = data.get('history', [])
                self.liked        = data.get('liked', [])
                self.disliked     = data.get('disliked', [])
                self.genre_counts = Counter(data.get('genre_counts', {}))

    def _save(self):
        with open(PROFILE_FILE, 'w') as f:
            json.dump({
                'history':      self.history,
                'liked':        self.liked,
                'disliked':     self.disliked,
                'genre_counts': dict(self.genre_counts)
            }, f)

    def add_watch(self, title, df):
        row = df[df['title'] == title]
        genres = row['genres_clean'].values[0] if len(row) else ''
        self.history = [x for x in self.history if x['title'] != title]
        self.history.insert(0, {
            'title': title, 'genres': genres,
            'time': datetime.now().strftime("%b %d, %H:%M")
        })
        for g in genres.split(): self.genre_counts[g] += 1
        self._save()

    def feedback(self, title, kind, df):
        if kind == 'like':
            if title not in self.liked: self.liked.append(title)
            if title in self.disliked:  self.disliked.remove(title)
            row = df[df['title'] == title]
            if len(row):
                for g in row['genres_clean'].values[0].split():
                    self.genre_counts[g] += 2
        else:
            if title not in self.disliked: self.disliked.append(title)
            if title in self.liked:         self.liked.remove(title)
        self._save()

    def top_genre(self):
        if not self.genre_counts: return None
        return self.genre_counts.most_common(1)[0][0]

    def genre_breakdown(self):
        if not self.genre_counts: return {}
        top = self.genre_counts.most_common(8)
        total = sum(v for _,v in top)
        return {k: round(v/total*100,1) for k,v in top}