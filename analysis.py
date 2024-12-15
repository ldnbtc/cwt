import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from textblob import TextBlob
from tqdm import tqdm

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

def extract_content_sections(entry):
    """
    Extracts different sections of the blog post content:
    - Tyler's commentary (excluding blockquotes)
    - Whether post ends with "Recommended"
    """
    content_div = entry.find('div', class_='entry-content')
    if not content_div:
        return "", False
    
    # Get all content elements
    elements = content_div.contents
    tyler_commentary = []
    in_blockquote = False
    
    for element in elements:
        if element.name == 'blockquote':
            in_blockquote = True
            continue
        elif in_blockquote and element.name in ['p', 'div']:
            in_blockquote = False
        
        if not in_blockquote and element.string:
            tyler_commentary.append(element.string.strip())
    
    # Join Tyler's commentary
    commentary = ' '.join(filter(None, tyler_commentary))
    
    # Check if post ends with "Recommended"
    ends_with_recommended = bool(re.search(r'Recommended\s*$', content_div.get_text().strip()))
    
    return commentary, ends_with_recommended

def extract_comment_count(entry):
    """
    Extract comment count from a blog post entry
    """
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
    
    return 0

def process_page(html, podcast_pattern, cutoff_date):
    """
    Process a single page of HTML and extract podcast posts
    """
    if not html:
        return [], False
    
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
        
        # Extract content sections
        commentary, ends_with_recommended = extract_content_sections(entry)
        
        qualifier_match = re.search(r'my\s+(\w+)\s+conversation', title, re.IGNORECASE)
        qualifier = qualifier_match.group(1) if qualifier_match else None
        
        # Extract comment count
        comment_count = extract_comment_count(entry)
        
        posts.append({
            'date': post_date,
            'title': title,
            'commentary': commentary,
            'url': entry.find('a')['href'] if entry.find('a') else '',
            'title_qualifier': qualifier,
            'comment_count': comment_count,
            'ends_with_recommended': ends_with_recommended
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
        return pd.DataFrame(columns=['date', 'title', 'qualifiers', 'sentiment_score', 'url', 
                                   'title_qualifier', 'comment_count', 'ends_with_recommended'])
    
    results = []
    
    for post in tqdm(posts, desc="Analyzing posts"):
        qualifiers = []
        if post.get('title_qualifier'):
            qualifiers.append(post['title_qualifier'].lower())
        
        # Look for qualifiers in title and Tyler's commentary
        if (re.search(r'excellent', post['title'], re.IGNORECASE) or 
            re.search(r'excellent', post['commentary'], re.IGNORECASE)):
            if 'excellent' not in qualifiers:
                qualifiers.append('excellent')
        
        other_qualifiers = ['fascinating', 'wonderful', 'great', 'outstanding', 'remarkable', 'contentious']
        for qualifier in other_qualifiers:
            if (re.search(rf'\b{qualifier}\b', post['title'], re.IGNORECASE) or 
                re.search(rf'\b{qualifier}\b', post['commentary'], re.IGNORECASE)):
                if qualifier not in qualifiers:
                    qualifiers.append(qualifier)
        
        # Sentiment analysis only on Tyler's commentary
        sentiment_score = TextBlob(post['commentary']).sentiment.polarity if post['commentary'] else 0
        
        results.append({
            'date': post['date'],
            'title': post['title'],
            'qualifiers': qualifiers,
            'sentiment_score': sentiment_score,
            'url': post['url'],
            'title_qualifier': post.get('title_qualifier'),
            'comment_count': post['comment_count'],
            'ends_with_recommended': post['ends_with_recommended']
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
        
        recommended_count = df['ends_with_recommended'].sum()
        print(f"\nPosts ending with 'Recommended': {recommended_count} ({(recommended_count/len(df)*100):.1f}%)")
        
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