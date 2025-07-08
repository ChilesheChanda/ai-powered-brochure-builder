import streamlit as st
import os
import requests
import json
import ollama
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

# Load API key
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key or not api_key.startswith('sk-'):
    st.error("Invalid OpenAI API key. Check your .env file.")
    st.stop()

USE_OPENAI = True  # Toggle here for OpenAI or Ollama
MODEL = 'gpt-4o-mini' if USE_OPENAI else 'llama3.2:3b'
openai_client = OpenAI(api_key=api_key) if USE_OPENAI else None

headers = {
    "User-Agent": "Mozilla/5.0"
}

class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        self.text = self.extract_text(soup)
        self.links = self.extract_links(soup)

    def extract_text(self, soup):
        if soup.body:
            for tag in soup.body(["script", "style", "img", "input"]):
                tag.decompose()
            return soup.body.get_text(separator="\n", strip=True)
        return ""

    def extract_links(self, soup):
        links = [link.get('href') for link in soup.find_all('a')]
        return [link for link in links if link and 'http' in link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

class LLMClient:
    def __init__(self, model=MODEL):
        self.model = model

    def get_relevant_links(self, website):
        link_system_prompt = """
        You are given a list of links from a company website.
        Select only relevant links for a brochure (About, Company, Careers, Products, Contact).
        Exclude login, terms, privacy, and emails.
        Return ONLY valid JSON like:
        {
            "links": [
                {"type": "about", "url": "https://company.com/about"},
                {"type": "contact", "url": "https://company.com/contact"}
            ]
        }
        """

        user_prompt = f"Links from {website.url}:\n{', '.join(website.links)}"

        if USE_OPENAI:
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": link_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return json.loads(response.choices[0].message.content.strip())
        else:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": link_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            result = response.get("message", {}).get("content", "").strip()
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"links": []}

    def generate_brochure(self, company_name, content, language):
        system_prompt = """
        You are a professional brochure writer. Write a fun, engaging, humorous company brochure in Markdown.
        """

        user_prompt = f"""
        Company: {company_name}
        Language: {language}
        Content:\n{content[:5000]}
        """

        if USE_OPENAI:
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        else:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.get("message", {}).get("content", "").strip()

class BrochureGenerator:
    def __init__(self, company_name, url, language='English'):
        self.company_name = company_name
        self.url = url
        self.language = language
        self.website = Website(url)
        self.llm_client = LLMClient()

    def generate(self):
        links = self.llm_client.get_relevant_links(self.website)
        content = self.website.get_contents()

        for link in links.get("links", []):
            try:
                linked_website = Website(link['url'])
                content += f"\n\n{link['type']}:\n{linked_website.get_contents()}"
            except:
                continue

        return self.llm_client.generate_brochure(self.company_name, content, self.language)

def main():
    st.title("📄 AI-powered Brochure Builder")

    st.markdown("""
💡 **AI-powered Brochure Builder** is your new digital copywriter! Just drop in a company name and website URL, and watch the magic happen. ✨  
It fetches the content, filters out the fluff, and crafts a fun, snappy brochure perfect for clients, investors, and future hires. 🎯

🔍 Using clever scraping (thanks, BeautifulSoup!) and smart language models from **OpenAI** or **Ollama** 🧠, it dives into pages like *About*, *Careers*, and *Products*, skipping the boring legal stuff. 🚫📄

🛠️ Built with cool tools like `OpenAI API`, `ollama`, and a touch of web scraping wizardry — it's made for devs with a bit of experience under their belt. 🧑‍💻💪

🚀 Whether you're automating content or just hate writing brochures (who doesn’t?), this app’s got your back.
""")


    company_name = st.text_input("Company Name", "Tour Eiffel")
    url = st.text_input("Website URL", "https://www.toureiffel.paris/fr")
    language = st.selectbox("Brochure Language", ["English", "French", "German", "Spanish"])

    if st.button("Generate Brochure"):
        with st.spinner("Scraping website and generating brochure..."):
            try:
                generator = BrochureGenerator(company_name, url, language)
                brochure = generator.generate()
                st.markdown(brochure)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
