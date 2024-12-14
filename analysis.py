import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from textblob import TextBlob
from tqdm import tqdm

def extract_comment_count(entry):
    """
    Extract comment count from a blog post entry
    """
    # First try to find a link containing "Comments"
    comment_text = None
    
    # Look for text ending in "Comments" or "Comment"
    for element in entry.strings:
        text = element.strip()
        if text.endswith('Comments') or text.endswith('Comment'):
            comment_text = text
            break
    
    if comment_text:
        # Extract the number from strings like "8 Comments"
        count_match = re.search(r'(\d+)\s+Comment', comment_text)
        if count_match:
            return int(count_match.group(1))
        elif comment_text == "Comment":  # Single comment case
            return 1
    
    # Backup method: look for comment links
    comment_link = entry.find('a', href=re.compile(r'#comment'))
    if comment_link:
        comment_text = comment_link.get_text().strip()
        count_match = re.search(r'(\d+)', comment_text)
        if count_match:
            return int(count_match.group(1))
    
    return 0

def fetch_page(url, headers):
    """
    Fetch a single page with error handling
    """
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None

def process_page(html, podcast_pattern, cutoff_date):
    """
    Process a single page of HTML and extract podcast posts
    """
    if not html:
        return []
    
    posts = []
    soup = BeautifulSoup(html, 'html.parser')
    entries = soup.find_all('article')
    
    for entry in entries:
        date_elem = entry.find('time')
        if not (date_elem and date_elem.get('datetime')):
            continue
            
        post_date = datetime.strptime(date_elem['datetime'][:10], '%Y-%m-%d')
        if post_date < cutoff_date:
            return posts, True  # Second value indicates we've hit the cutoff
        
        title_elem = entry.find('h1') or entry.find('h2')
        if not title_elem:
            continue
            
        title = title_elem.text.strip()
        if not podcast_pattern.search(title):
            continue
            
        content = entry.find('div', class_='entry-content')
        content_text = content.text.strip() if content else ''
        
        qualifier_match = re.search(r'my\s+(\w+)\s+conversation', title, re.IGNORECASE)
        qualifier = qualifier_match.group(1) if qualifier_match else None
        
        # Extract comment count
        comment_count = extract_comment_count(entry)
        
        posts.append({
            'date': post_date,
            'title': title,
            'content': content_text,
            'url': entry.find('a')['href'] if entry.find('a') else '',
            'title_qualifier': qualifier,
            'comment_count': comment_count
        })
    
    return posts, False

def get_blog_posts(base_url, max_pages=125):
    """
    Scrapes blog posts from the website starting from January 1, 2024
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    podcast_pattern = re.compile(r'my(?:\s+\w+)?\s+conversation(?:\s+(?:is|with))?|conversation(?:\s+(?:is|with))?', re.IGNORECASE)
    cutoff_date = datetime(2024, 1, 1)
    all_posts = []
    
    urls = [f"{base_url}/page/{page}" if page > 1 else base_url for page in range(1, max_pages + 1)]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(fetch_page, url, headers): url for url in urls}
        
        with tqdm(total=len(urls), desc="Fetching pages") as pbar:
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    html = future.result()
                    if html:
                        posts, reached_cutoff = process_page(html, podcast_pattern, cutoff_date)
                        all_posts.extend(posts)
                        if reached_cutoff:
                            break
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                finally:
                    pbar.update(1)
                    time.sleep(0.5)
    
    return all_posts

def analyze_posts(posts):
    """
    Analyzes posts for qualifiers and sentiment
    """
    if not posts:
        print("No posts to analyze.")
        return pd.DataFrame(columns=['date', 'title', 'qualifiers', 'sentiment_score', 'url', 'title_qualifier', 'comment_count'])
    
    results = []
    
    for post in tqdm(posts, desc="Analyzing posts"):
        qualifiers = []
        if post.get('title_qualifier'):
            qualifiers.append(post['title_qualifier'].lower())
        
        if (re.search(r'excellent', post['title'], re.IGNORECASE) or 
            re.search(r'excellent', post['content'], re.IGNORECASE)):
            if 'excellent' not in qualifiers:
                qualifiers.append('excellent')
        
        other_qualifiers = ['fascinating', 'wonderful', 'great', 'outstanding', 'remarkable', 'contentious']
        for qualifier in other_qualifiers:
            if (re.search(rf'\b{qualifier}\b', post['title'], re.IGNORECASE) or 
                re.search(rf'\b{qualifier}\b', post['content'], re.IGNORECASE)):
                if qualifier not in qualifiers:
                    qualifiers.append(qualifier)
        
        blob = TextBlob(post['content'])
        
        results.append({
            'date': post['date'],
            'title': post['title'],
            'qualifiers': qualifiers,
            'sentiment_score': blob.sentiment.polarity,
            'url': post['url'],
            'title_qualifier': post.get('title_qualifier'),
            'comment_count': post['comment_count']
        })
    
    return pd.DataFrame(results)

def main():
    base_url = 'https://marginalrevolution.com'
    
    print("Fetching blog posts...")
    posts = get_blog_posts(base_url, max_pages=125)
    
    print(f"\nFound {len(posts)} podcast-related posts since January 1, 2024")
    
    print("\nAnalyzing posts...")
    df = analyze_posts(posts)
    
    if len(df) > 0:
        print("\nPosts by title qualifier:")
        title_qualifiers = df[df['title_qualifier'].notna()]['title_qualifier'].value_counts()
        print(title_qualifiers)
        
        print("\nPosts with 'excellent' qualifier:")
        excellent_posts = df[df['qualifiers'].apply(lambda x: 'excellent' in x)]
        print(f"Total: {len(excellent_posts)}")
        for _, post in excellent_posts.iterrows():
            print(f"- {post['date'].strftime('%Y-%m-%d')}: {post['title']} ({post['comment_count']} comments)")
        
        print("\nAll qualifiers found:")
        all_qualifiers = [q for quals in df['qualifiers'] for q in quals]
        if all_qualifiers:
            qualifier_counts = pd.Series(all_qualifiers).value_counts()
            print(qualifier_counts)
        else:
            print("No qualifiers found")
        
        print("\nAverage sentiment score:", df['sentiment_score'].mean())
        print("Average comment count:", df['comment_count'].mean())
        
        # Save results to CSV
        df.to_csv('podcast_analysis.csv', index=False)
        print("\nResults saved to podcast_analysis.csv")
    else:
        print("No posts found to analyze.")

if __name__ == "__main__":
    main()