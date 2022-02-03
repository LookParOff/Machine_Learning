"""
this script parse site pikabu.com. Output result is a table with columns
post-name, post-link, post-tags, post-time
"""
import requests  # sending request on server
from fake_useragent import UserAgent  # for pretending be a user
from bs4 import BeautifulSoup  # for parsing and find a html tags and attributes
import pandas as pd
import time
import socks
import socket
import seaborn as sns
import matplotlib.pyplot as plt


def checkIP():
    ip = requests.get('http://checkip.dyndns.org').content
    soup = BeautifulSoup(ip, 'html.parser')
    return soup.find('body').text


def get_data_frame_of_site(sleep_secs, count_of_pages, start_page=0):
    """
    returns the table of site's data
    :param sleep_secs:
    :param count_of_pages:
    :param start_page:
    :return:
    """
    for num_page in range(count_of_pages):
        try:
            page_link = f'https://pikabu.ru/tag/Собака?page={start_page + num_page + 1}'
            response = requests.get(page_link, headers={'User-Agent': UserAgent().chrome})
            print("page №", num_page, response, end=" ")
            # ip = checkIP()
            if not response.ok:
                time.sleep(30)
                response = requests.get(page_link, headers={'User-Agent': UserAgent().chrome})
                if not response.ok:
                    print("We still blocked. We are fucked! Abort loop!")
                    break
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            links_html = soup.findAll("a", attrs={"class": "story__title-link"})
            tags_html = soup.findAll("div", attrs={"class": "story__tags"})
            nicks_html = soup.findAll("a", attrs={"class": "story__user-link user__nick"})
            times_html = soup.findAll("time", {"class": "caption story__datetime hint"})

            for i, nick in enumerate(nicks_html):
                if nick.attrs["data-name"] == "specials":
                    links_html.pop(i)
                    # times_html.pop(i)

            print(len(links_html) == len(tags_html) == len(times_html), len(times_html), end=" ")

            for link, tag, post_time in zip(links_html, tags_html, times_html):
                data.append([])
                data[-1].append(next(link.children))  # Title
                data[-1].append(link.attrs["href"])  # Link to post
                data[-1].append([t.attrs["data-tag"]
                                 for t in tag.findAll("a", attrs={"class": "tags__tag"})])  # Tags
                data[-1].append(post_time.attrs["datetime"])
        except:
            print("Oh no, error has occurred. Anyway, go to the next page.")

        time.sleep(sleep_secs)
        print()
    data_frame = pd.DataFrame(data)
    return data_frame


def load_df():
    """
    loads already parsed table and plot the graph of posts
    :return:
    """
    df = pd.read_csv(r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\pikabu\dogs_posts 100 pages.csv",
                     sep=";")
    df.columns = ["Name", "Link", "Tags", "Post-time"]
    df.iloc[:, 3] = pd.to_datetime(df.iloc[:, 3])
    df.iloc[:, 3] = df.iloc[:, 3].dt.date
    print(df)
    # df_ = df.iloc[:, [0, 3]]
    grouped_df = df.groupby("Post-time").count()
    sns.relplot(data=grouped_df, x="Post-time", y="Name", kind="line")
    plt.show()


if __name__ == "__main__":
    # print(checkIP())
    # socks.set_default_proxy(socks.SOCKS5, "localhost", 9050)  # if we wanna use tor ips)
    # socket.socket = socks.socksocket
    print(checkIP())
    data = []
    df = get_data_frame_of_site(10, 5)
    df.replace(";", ".", inplace=True)
    df.to_csv(r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\pikabu\dogs_posts4000.csv",
              sep=";", encoding="UTF-8")
