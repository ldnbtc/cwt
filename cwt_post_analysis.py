import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import re

def get_blog_posts(base_url):
    """
    Scrapes blog posts from the website starting from January 1, 2023
    """
    posts = []
    page = 1
    cutoff_date = datetime(2023, 1, 1)
    
    while True:
        url = f"{base_url}/page/{page}" if page > 1 else base_url
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all blog post entries
        entries = soup.find_all('article')
        
        if not entries:
            break
            
        for entry in entries:
            # Extract date
            date_elem = entry.find('time')
            if date_elem and date_elem.get('datetime'):
                post_date = datetime.strptime(date_elem['datetime'][:10], '%Y-%m-%d')
                
                # Stop if we've reached posts before 2023
                if post_date < cutoff_date:
                    return posts
                
                # Extract title and content
                title = entry.find('h1').text.strip() if entry.find('h1') else ''
                content = entry.find('div', class_='entry-content').text.strip() if entry.find('div', class_='entry-content') else ''
                
                # Check if it's a podcast post
                if 'my conversation with' in title.lower():
                    posts.append({
                        'date': post_date,
                        'title': title,
                        'content': content,
                        'url': entry.find('a')['href'] if entry.find('a') else ''
                    })
        
        page += 1
    
    return posts

def analyze_posts(posts):
    """
    Analyzes posts for qualifiers and sentiment
    """
    results = []
    
    for post in posts:
        # Look for qualifiers
        qualifiers = []
        if 'excellent' in post['title'].lower() or 'excellent' in post['content'].lower():
            qualifiers.append('excellent')
            
        # Look for other common positive qualifiers
        other_qualifiers = ['fascinating', 'wonderful', 'great', 'outstanding', 'remarkable']
        for qualifier in other_qualifiers:
            if qualifier in post['title'].lower() or qualifier in post['content'].lower():
                qualifiers.append(qualifier)
                
        # Sentiment analysis using TextBlob
        blob = TextBlob(post['content'])
        sentiment = blob.sentiment.polarity
        
        results.append({
            'date': post['date'],
            'title': post['title'],
            'qualifiers': qualifiers,
            'sentiment_score': sentiment,
            'url': post['url']
        })
    
    return results

def main():
    base_url = 'https://marginalrevolution.com'
    
    print("Fetching blog posts...")
    posts = get_blog_posts(base_url)
    
    print(f"\nFound {len(posts)} podcast-related posts since January 1, 2023")
    
    print("\nAnalyzing posts...")
    results = analyze_posts(posts)
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    print("\nPosts with 'excellent' qualifier:")
    excellent_posts = df[df['qualifiers'].apply(lambda x: 'excellent' in x)]
    print(f"Total: {len(excellent_posts)}")
    for _, post in excellent_posts.iterrows():
        print(f"- {post['date'].strftime('%Y-%m-%d')}: {post['title']}")
    
    print("\nOther qualifiers found:")
    all_qualifiers = [q for quals in df['qualifiers'] for q in quals]
    qualifier_counts = pd.Series(all_qualifiers).value_counts()
    print(qualifier_counts)
    
    print("\nAverage sentiment score:", df['sentiment_score'].mean())
    
    # Save results to CSV
    df.to_csv('podcast_analysis.csv', index=False)
    print("\nResults saved to podcast_analysis.csv")

if __name__ == "__main__":
    main()