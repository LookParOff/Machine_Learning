"""
this script parse site pikabu.com. Output result is a table with columns
post-name, post-link, post-tags, post-time
"""
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
import socks
import socket



def get_data_frame_of_site(sleep_secs, count_of_pages, start_page=0):
    """
    returns the table of site's data
    :param sleep_secs:
    :param count_of_pages:
    :param start_page:
    :return:
    """
    added_posts = set()

    for page_num in range(count_of_pages):
        try:
            posts = driver.find_elements(By.XPATH, "//div[@class='story__main']")
            swipe_sides = driver.find_elements(By.XPATH, "//div[@class='story__left ']")
            print("before change", len(posts), len(swipe_sides))
            displayed_posts = [p.find_element(By.CLASS_NAME, "story__title-link").get_attribute("href")
                               not in added_posts for p in posts]
            start_ind = displayed_posts.index(True)
            posts = posts[start_ind:]
            swipe_sides = swipe_sides[start_ind:]
            print("after change", len(posts), len(swipe_sides))
            for post, swipe_side in zip(posts, swipe_sides):
                nick = post.find_element(By.CLASS_NAME,
                                         "story__user-link.user__nick").get_attribute("data-name")
                if nick == "specials":
                    continue
                try:
                    views = post.find_element(By.CLASS_NAME, "story__views.hint").text
                except selenium.common.exceptions.NoSuchElementException:
                    views = 0
                title = post.find_element(By.CLASS_NAME, "story__title-link").text
                print(title)
                link = post.find_element(By.CLASS_NAME, "story__title-link").get_attribute("href")
                post_id = link[link.rfind("_") + 1:]
                tags = post.find_element(By.CLASS_NAME, "story__tags.tags").text
                post_time = post.find_element(By.TAG_NAME, "time").get_attribute("datetime")
                count_comments = post.find_element(By.CLASS_NAME,
                                                         "story__comments-link-count").text
                rating = swipe_side.find_element(By.CLASS_NAME, "story__rating-count").text
                data.append([post_id, title, nick,
                             views, rating, count_comments,
                             tags, link, post_time])
                added_posts.add(link)

                time.sleep(random.randint(sleep_secs[0], sleep_secs[1]))
                swipe_side.click()
                time.sleep(0.2)  # just little wait in case of loading something
        except:
            print("Fuck! Something goes wrong")
            driver.save_screenshot(f"D:\\{time.asctime()}_{page_num}_{len(data)}.png")
        print("We end one page! Now I wait some for loading")
        time.sleep(3)

    data_frame = pd.DataFrame(data)
    return data_frame


def load_df():
    """
    loads already parsed table and plot the graph of posts
    :return:
    """
    df = pd.read_csv(
        r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\pikabu\dogs_posts 100 pages.csv",
        sep=";")
    df.columns = ["post_id", "title", "nick",
                  "views", "rating", "count_comments",
                  "tags", "link", "post_time"]
    df.iloc[:, 3] = pd.to_datetime(df.iloc[:, 3])
    df.iloc[:, 3] = df.iloc[:, 3].dt.date
    print(df)
    # df_ = df.iloc[:, [0, 3]]
    grouped_df = df.groupby("post_time").count()
    sns.relplot(data=grouped_df, x="post_time", y="title", kind="line")
    plt.show()


if __name__ == "__main__":
    ser = Service(r"C:\Program Files (x86)\Google\chromedriver.exe")
    op = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=ser, options=op)
    driver.set_window_position(-1200, 0)
    driver.maximize_window()
    page_link = f'https://pikabu.ru/tag/Собака'
    driver.get(page_link)
    sign_in = driver.find_elements(By.TAG_NAME, "form")[1]  # this is reference to sign in form
    login, password = sign_in.find_elements(By.CLASS_NAME, "input__input")
    login.send_keys("Nickname")
    password.send_keys("password")  # TODO change password :)
    time.sleep(3)
    driver.find_element(By.CLASS_NAME, "button_success.button_width_100").click()  # click submit
    time.sleep(10)  # maybe need insert captcha
    data = []
    df = get_data_frame_of_site((1, 7), 100)  # 250 pages is 5000 posts
    driver.quit()

    df.replace(";", ".", inplace=True)
    df.to_csv(r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\pikabu\dogs_posts4000.csv",
              sep=";", encoding="UTF-8")
