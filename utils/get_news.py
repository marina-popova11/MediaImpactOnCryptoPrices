import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import base64

BASE_KEYWORDS = [
    "bitcoin", "BTC", "cryptocurrency", "crypto",
    "blockchain", "digital currency", "virtual currency",
    "token", "altcoin", "stablecoin"
]

TECH_KEYWORDS = [
    "mining", "staking", "halving", "currency exchange rate", "liquidity",
    "phishing", "burning tokens"
]

POLITIC_KEYWORDS = [
    "inflation", "elections", "high tariffs", "legalization of digital currencies", "cryptocurrency laws", "pandemics", "military conflicts"
]

EVENT_KEYWORDS = [
    "growth", "fall", "collapse", "takeoff",
    "regulation", "ban", "investment", "trading",
    "wallet", "exchange", "bullrun", "crash"
]

ENTITY_KEYWORDS = [
    "Coinbase", "CoinGecko", "Elon Musk", "Investment", "trust", "Donald Trump"
]

KEYWORD_GROUPS = {
    "base": BASE_KEYWORDS,
    "tech": TECH_KEYWORDS,
    "politic": POLITIC_KEYWORDS,
    "event": EVENT_KEYWORDS,
    "entity": ENTITY_KEYWORDS
}

URL = "https://min-api.cryptocompare.com/data/v2/news/"

def fetch_cryptocompare_news(query: dict, language: str = "EN",
                             start_date="2025-10-01", end_date="2025-10-31",
                             sleep_time=1):
    all_keywords = [word.lower() for words in query.values() for word in words]
    all_posts = []

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    params = {
        "lang": language,
        "categories": "Blockchain,Altcoin,Crypto",
        "api_key": API_KEY
    }

    try:
        response = requests.get(URL, params=params, timeout=15)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text[:150]}")
            return pd.DataFrame()

        data = response.json()
        if "Data" not in data:
            print("Error API:", data)
            return pd.DataFrame()

        articles = data["Data"]

        for art in articles:
            title = art.get("title", "")
            body = art.get("body", "")
            published_ts = art.get("published_on", 0)
            published_dt = pd.to_datetime(published_ts, unit="s")

            if not (start <= published_dt <= end):
                continue

            text = f"{title}. {body}".lower()
            if not any(kw in text for kw in all_keywords):
                continue

            article = {
                "title": title,
                "description": body,
                "publishedAt": published_dt,
                "source": art.get("source", "CryptoCompare"),
                "text": f"{title}. {body}"
            }
            all_posts.append(article)

        time.sleep(sleep_time)

    except Exception as e:
        print("Error fetching CryptoCompare news:", e)
        return pd.DataFrame()

    if all_posts:
        df = pd.DataFrame(all_posts)
        df.drop_duplicates(subset=["text"], inplace=True)
        df["publishedAt"] = pd.to_datetime(df["publishedAt"]).dt.tz_localize(None)
        df["window"] = df["publishedAt"].dt.floor("1h")
        df = df.sort_values("publishedAt")
        print(len(df), "news found")
        return df
    else:
        print("No news found")
        return pd.DataFrame()

CLIENT_ID = ""
CLIENT_SECRET = ""
USER_AGENT = "CryptoNewsAnalyzer"

def get_reddit_access_token():
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')

    headers = {
        'User-Agent': USER_AGENT,
        'Authorization': f'Basic {auth_base64}'
    }

    data = {
        'grant_type': 'client_credentials',
        'device_id': 'DO_NOT_TRACK_THIS_DEVICE'
    }

    try:
        response = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            headers=headers,
            data=data,
            timeout=10
        )

        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data['access_token']
            print("Successfully obtained Reddit OAuth token")
            return access_token
        else:
            print(f"Failed to get OAuth token: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error getting OAuth token: {e}")
        return None


def fetch_reddit_news(query: dict, start_date="2025-10-01", end_date="2025-10-31", subreddits=None,
                      posts_per_keyword=50):
    access_token = get_reddit_access_token()
    if not access_token:
        print("Failed to get Reddit OAuth token")
        return pd.DataFrame()

    crypto_subreddits = [
        "CryptoCurrency",
        "CryptoMarkets",
        "bitcoin",
        "blockchain",
        "altcoin",
        "defi",
        "CryptoTechnology"
    ]

    if subreddits is None:
        subreddits = crypto_subreddits

    all_posts = []
    all_keywords = [word for words in query.values() for word in words]
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    headers = {
        'User-Agent': USER_AGENT,
        'Authorization': f'Bearer {access_token}'
    }

    for word in all_keywords:
        for i, subreddit in enumerate(subreddits):
            print(f"search {subreddit} for {word}")
            url = f"https://oauth.reddit.com/r/{subreddit}/search"
            params = {
                'q': word,
                'limit': posts_per_keyword,
                'restrict_sr': 'on',
                'sort': 'new',
                'type': 'link,self'
            }

            try:
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 401:
                    print("Token expired. Refreshing...")
                    access_token = get_reddit_access_token()
                    if access_token:
                        headers['Authorization'] = f'Bearer {access_token}'
                        continue
                    else:
                        break

                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 30))
                    print(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue

                elif response.status_code != 200:
                    print(f"Error {response.status_code} for r/{subreddit}")
                    continue

                data = response.json()
                if 'data' not in data or 'children' not in data['data']:
                    print(f"No data for {subreddit}")
                    continue

                posts = data.get("data", {}).get("children", [])
                posts_count = 0
                for post in posts:
                    post_data = post['data']

                    if post_data.get('removed_by_category'):
                        continue

                    created_utc = post_data.get('created_utc', 0)
                    published_at = datetime.fromtimestamp(created_utc)

                    if published_at < start or published_at > end:
                        continue

                    title = post_data.get("title", "") or ""
                    body = post_data.get("selftext", "") or ""
                    text = f"{title}. {body}".lower()

                    matched_keywords = [word for word in all_keywords if word.lower() in text]
                    if not matched_keywords:
                        continue

                    article = {
                        'title': title,
                        'description': body[:500],
                        'publishedAt': published_at,
                        'upvotes': post_data.get('ups', 0),
                        'comments': post_data.get('num_comments', 0),
                        'subreddit': subreddit,
                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                        'score': post_data.get('score', 0)
                    }

                    if (len(article['title']) > 10 and
                            article['upvotes'] >= 10 and article['comments'] > 5):
                        all_posts.append(article)
                        posts_count += 1

                print(f"Found {posts_count} quality posts in r/{subreddit}")
            except Exception as e:
                print(f"Error getting posts: {e}")

            if i < len(crypto_subreddits) - 1:
                time.sleep(1)

    if all_posts:
        df = pd.DataFrame(all_posts)
        df["text"] = df[["title", "description"]].astype(str).agg(". ".join, axis=1)
        df["publishedAt"] = pd.to_datetime(df["publishedAt"]).dt.tz_localize(None)
        df.dropna(subset=["text", "publishedAt"], inplace=True)
        df["window"] = df["publishedAt"].dt.floor("1h")
        df = df.sort_values('publishedAt')
        print(len(df))
        return df
    else:
        return pd.DataFrame()

API_KEY_NEWS_API = ""

def fetch_newsapi_news(query: dict, start_date="2025-10-01", end_date="2025-10-31", language="en", posts_per_keyword=50,
                       sleep_time=2):
    all_posts = []
    all_keywords = [word for words in query.values() for word in words]
    url = "https://newsapi.org/v2/everything"
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    for word in all_keywords:
        print(f"search for {word}")
        current = start
        while current <= end:
            day_start = current.strftime("%Y-%m-%d")
            day_end = (current + delta).strftime("%Y-%m-%d")
            params = {
                "q": word,
                "from": day_start,
                "to": day_end,
                "language": language,
                "sortBy": "publishedAt",
                "pageSize": posts_per_keyword,
                "apiKey": API_KEY_NEWS_API
            }

            try:
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 429:
                    print("Rate limit reached. Sleeping for 60 seconds...")
                    time.sleep(60)
                    continue
                elif response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text[:150]}")
                    current += delta
                    continue

                data = response.json()
                if "articles" not in data or not data["articles"]:
                    print(f"No articles for {word}")
                    current += delta
                    continue

                articles = data["articles"]
                count = 0
                for art in articles:
                    title = art.get("title", "") or ""
                    description = art.get("description", "") or ""
                    content = art.get("content", "") or ""
                    text = f"{title}. {description}. {content}".lower()
                    matched_keywords = [kw for kw in all_keywords if kw.lower() in text]
                    if not matched_keywords:
                        continue

                    published_at = art.get("publishedAt")
                    if not published_at:
                        continue
                    published_dt = pd.to_datetime(published_at, utc=True).tz_convert(None)
                    article = {
                        "title": title,
                        "description": description,
                        "publishedAt": published_dt,
                        "source": art.get("source", {}).get("name", ""),
                        "url": art.get("url", ""),
                        "text": f"{title}. {description}. {content}"
                    }
                    all_posts.append(article)
                    count += 1

                print(f"Found {count} relevant articles for '{word}'")
            except Exception as e:
                print(f"Error fetching for '{word}': {e}")
            current += delta

            time.sleep(sleep_time)

    if all_posts:
        df = pd.DataFrame(all_posts)
        df["publishedAt"] = pd.to_datetime(df["publishedAt"]).dt.tz_localize(None)
        df.dropna(subset=["text", "publishedAt"], inplace=True)
        df["window"] = df["publishedAt"].dt.floor("1h")
        df = df.sort_values("publishedAt")
        print(len(df))
        return df
    else:
        print("No articles found.")
        return pd.DataFrame()