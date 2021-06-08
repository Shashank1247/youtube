from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import demoji
from langdetect import detect
import re   # regular expression
import pandas as pd
import os
import auth



os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
service = auth.service
print(service)

query = "Taarak Mehta Ka Ooltah Chashmah - Ep 2738 - Full Episode - 24th May, 2019"

query_results = service.search().list(part = 'snippet',q = query,
                                      order = 'relevance', 
                                      type = 'video',
                                      relevanceLanguage = 'en',
                                      safeSearch = 'moderate').execute()

video_id = []
channel = []
video_title = []
video_desc = []
for item in query_results['items']:
    video_id.append(item['id']['videoId'])
    channel.append(item['snippet']['channelTitle'])
    video_title.append(item['snippet']['title'])
    video_desc.append(item['snippet']['description'])

video_id = video_id[0]
channel = channel[0]
video_title = video_title[0]
video_desc = video_desc[0]

print(video_id[1])