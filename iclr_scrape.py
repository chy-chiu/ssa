# # %%
# import requests
# from bs4 import BeautifulSoup, BeautifulStoneSoup

# r = requests.get("https://openreview.net/group?id=ICLR.cc/2026/Conference#tab-active-submissions")
# r.content
# # %%
# soup = BeautifulSoup(r.content, "html.parser")
# soup
# # %%
# from requests_html import HTMLSession

# session = HTMLSession()
# r = session.get("https://openreview.net/group?id=ICLR.cc/2026/Conference#tab-active-submissions")

# # render() modifies r.html in place and returns None
# r.html.render(timeout=30, sleep=2)  # wait for JS to load

# # Now access the rendered content
# print(r.html.html[:1000])  # print first 1000 chars


# # %%
# import requests

# # Get submissions directly from API
# url = "https://api2.openreview.net/notes"
# params = {
#     'invitation': 'ICLR.cc/2026/Conference/-/Submission',
#     'details': 'replyCount,invitation',
#     'limit': 1000
# }

# response = requests.get(url, params=params)
# print(response)
# data = response.json()

# print(f"Found {len(data['notes'])} submissions")
# for note in data['notes'][:5]:  # first 5
#     print(f"\nTitle: {note['content']['title']['value']}")
#     print(f"Authors: {note['content']['authors']['value']}")

# # %%
# # API V2


# %%
import openreview
import csv
import sys
import argparse

client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username="chy.chiu@gmail.com",
    password="alamakAI6*"
)
venue_id = "ICLR.cc/2026/Conference"
venue_group = client.get_group(venue_id)

submission_name = venue_group.content['submission_name']['value']
submissions = client.get_all_notes(invitation=f'{venue_id}/-/{submission_name}')
# %%
len(submissions)
# %%
ratings_name = venue_group.content['review_rating']['value']
ratings = client.get_all_notes(invitation=f'{venue_id}/-/{ratings_name}')
# %%
len(ratings)
# %%

ratings_name

# %%
