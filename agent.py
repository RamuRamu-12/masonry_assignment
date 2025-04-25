import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from dotenv import load_dotenv
import streamlit as st
from typing import Any, Dict, List, Optional

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish

load_dotenv()

# LLM Setup with OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ===================
# Custom Callback Handler
# ===================
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, st_container):
        self.st_container = st_container
        self.thinking_expander = None
        self.current_step = 1
    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        tool_name = action.tool
        tool_input = action.tool_input
        
        if not self.thinking_expander:
            self.thinking_expander = self.st_container.expander("Agent Thinking Process", expanded=True)
        
        with self.thinking_expander:
            st.markdown(f"### Step {self.current_step}: Using {tool_name}")
            st.markdown(f"**Tool Input:**")
            st.code(tool_input)
            self.current_step += 1
    
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        if self.thinking_expander:
            with self.thinking_expander:
                st.markdown("**Tool Output:**")
                st.code(output[:2000] + ("..." if len(output) > 2000 else ""))  # Limit output display
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        if self.thinking_expander:
            with self.thinking_expander:
                st.success("Agent completed its task!")

# ===================
# Tool 1: Query Analyzer
# ===================
query_analysis_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an expert research strategist. Carefully analyze the user's query:

1. Identify the research intent (fact-finding, trend analysis, news investigation, etc.)
2. Determine the exact type of information needed (facts, recent events, stats, opinions, etc.)
3. Break down the question into detailed searchable components.
4. Propose a strong multi-step search strategy for maximum relevance.

Query:
{query}

Format the output clearly with these sections:
##### Intent
##### Information Type
##### Searchable Components
##### Search Strategy
"""
)

def analyze_query(query: str) -> str:
    return llm.invoke(query_analysis_prompt.format(query=query))

# ===================
# Tool 2: Bing Search (mocked)
# ===================
def get_news_urls(query: str, limit: int = 10, pages: int = 3):
    headers = {"User-Agent": "Mozilla/5.0"}
    all_links = []
    try:
        for i in range(pages):
            offset = i * 10
            url = f"https://www.bing.com/search?q={query}&first={offset}"
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("li", class_="b_algo")
            for result in results:
                a = result.find("a")
                if a and a.get("href"):
                    all_links.append(a["href"])
            if len(all_links) >= limit:
                break
        return all_links[:limit]
    except Exception:
        return []

# ===================
# Tool 3: Web Scraper
# ===================
def is_scraping_allowed(url: str) -> bool:
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    rp = RobotFileParser()
    try:
        rp.set_url(f"{base_url}/robots.txt")
        rp.read()
        return rp.can_fetch("Mozilla/5.0", url)
    except:
        return True

def extract_article_content(url: str) -> str:
    if not is_scraping_allowed(url):
        return f"Scraping not allowed by site's robots.txt: {url}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        texts = []
        for tag in ['p', 'li', 'td']:
            texts.extend([el.get_text(strip=True) for el in soup.find_all(tag)])
        return "\n".join(texts)[:3000] if texts else "No readable content found."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# ===================
# Tool 4: News Synthesizer
# ===================
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
You are a synthesis expert. Combine insights from the extracted content:

- Identify common facts and flag any contradictions
- Structure clearly: events, causes, impacts, people, timeline
- Output in markdown bullet format followed by references

Content:
{content}
"""
)

def realtime_news_summary_tool(query: str) -> str:
    search_query = f"{query} site:news.google.com OR site:bbc.com OR site:reuters.com"
    urls = get_news_urls(search_query, limit=10, pages=3)
    if not urls:
        return "No news articles found."

    all_content = ""
    refs = []
    for url in urls:
        content = extract_article_content(url)
        if content and "not allowed" not in content:
            all_content += content + "\n\n"
            refs.append(url)

    summary = llm.invoke(summary_prompt.format(content=all_content[:3000]))
    return summary + "\n\n" + "\n".join([f"🔗 [Source]({u})" for u in refs])

# ===================
# Tool 5: Content Analyzer
# ===================
content_analyzer_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
You are a content analyst. Examine the text below:
- Identify key facts and statistics
- Highlight takeaways and contradictions
- Summarize in markdown bullet points

{content}
"""
)

def content_analyzer_tool(content: str) -> str:
    return llm.invoke(content_analyzer_prompt.format(content=content))

# ===================
# Tool 6: Keyword Extractor
# ===================
keyword_extractor_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
Extract the top 10 high-value keywords or topics from the content below. Output as a comma-separated list:
display the {content} also in the output.
{content}
"""
)

def keyword_extractor_tool(content: str) -> str:
    return llm.invoke(keyword_extractor_prompt.format(content=content))

# ===================
# Tool Registry
# ===================
tools = [
    Tool(name="AnalyzeQuery", func=analyze_query, description="Analyze a query and break it down."),
    Tool(name="SearchNews", func=lambda q: "\n".join(get_news_urls(q)), description="Perform Bing-based news search and return URLs."),
    Tool(name="ScrapeURL", func=extract_article_content, description="Extract content from a web URL respecting robots.txt."),
    Tool(name="SummarizeNews", func=realtime_news_summary_tool, description="Summarize multiple articles into key insights."),
    Tool(name="AnalyzeContent", func=content_analyzer_tool, description="Analyze given content and produce insights."),
    Tool(name="ExtractKeywords", func=keyword_extractor_tool, description="Get top keywords from a block of content.")
]

# ===================
# Streamlit UI
# ===================
st.set_page_config(page_title="Web Research Agent (OpenAI)", layout="centered")

st.title("🔍 AI Web Research Agent (OpenAI)")
st.write("Input a research query. The agent will analyze, search, scrape, and synthesize information.")

query = st.text_area("Enter your research query:", height=150)

if st.button("Run Research Agent"):
    process_container = st.container()
    result_container = st.container()
    
    callback_handler = StreamlitCallbackHandler(process_container)

    with st.spinner("Running multi-tool agent..."):
        # Create a guided prompt that instructs the agent to use each tool in sequence
        guided_prompt = f"""
        Please follow this multi-step process strictly using the provided tools:

        1. Use **AnalyzeQuery** on the following query: "{query}"
        2. Use **SearchNews** with the same query to find relevant news article URLs.
        3. For each URL found, use **ScrapeURL** to extract article content.
        4. Then use **SummarizeNews** to summarize the combined content from all URLs.
        5. Use **AnalyzeContent** on the combined scraped content to extract insights.
        6. Finally, use **ExtractKeywords** to list top topics from the scraped content.

        Proceed step-by-step using the tools.
        """

        try:
            agent = initialize_agent(
                tools=tools, 
                llm=llm, 
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                verbose=True
            )
            
            result = agent.run(guided_prompt, callbacks=[callback_handler])

            with result_container:
                st.markdown("### Final Output")
                st.markdown(result)
        except Exception as e:
            st.error(f"Error: {str(e)}")
