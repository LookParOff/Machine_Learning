"""
this script parse site pikabu.com. Output result is a table with columns
post-name, post-link, post-tags, post-time
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time
import random
import socks
import socket


def wait(a, b):
    t = random.uniform(a, b)
    time.sleep(t)


def load_page_by_date(start_date, end_date):
    """load posts in range of start_date and end_date"""
    driver.execute_script("window.scrollTo(0, 0)")
    wait(0.5, 1.5)
    btn = driver.find_element(By.XPATH, "//div[@class='form-group stories-search__dates-group']")
    btn.click()  # click to choose search by date
    wait(0.5, 1.5)
    btn = driver.find_element(By.XPATH, "//span[contains(@data-value, 'range')]")
    btn.click()  # click to get accesses to calendar
    wait(0.5, 1.5)
    inp_box_from = driver.find_element(By.XPATH, "//input[contains(@data-type, 'from')]")
    inp_box_from.click()
    wait(0.5, 1.5)
    inp_box_from.send_keys(Keys.CONTROL + "a")
    wait(0.5, 1.5)
    inp_box_from.send_keys(Keys.DELETE)
    wait(0.5, 1.5)
    inp_box_from.send_keys(start_date)
    wait(0.5, 1.5)
    inp_box_to = driver.find_element(By.XPATH, "//input[contains(@data-type, 'to')]")
    inp_box_to.click()
    wait(0.5, 1.5)
    inp_box_to.send_keys(Keys.CONTROL + "a")
    wait(0.5, 1.5)
    inp_box_to.send_keys(Keys.DELETE)
    wait(0.5, 1.5)
    inp_box_to.send_keys(end_date)
    wait(0.5, 1.5)
    driver.find_element(By.XPATH, "//div[@class='calendar-head']").click()  # just for accept date
    wait(0.5, 1.5)
    driver.execute_script("window.scrollTo(0, 450)")
    wait(0.5, 1.5)
    driver.find_element(By.XPATH,
                        "//div[@class='form-group stories-search__dates-group form-group_open']//"
                        "div[@class='form-group__button']").click()  # close calendar
    wait(3, 4)  # wait for loading


def parse_post(post, swipe_side):
    link = post.find_element(By.CLASS_NAME, "story__title-link").get_attribute("href")
    nick = post.find_element(By.CLASS_NAME,
                             "story__user-link.user__nick").get_attribute("data-name")
    if nick == "specials":
        return link
    try:  # in case there are no views
        views = post.find_element(By.CLASS_NAME, "story__views.hint").text
    except NoSuchElementException:
        views = 0
    title = post.find_element(By.CLASS_NAME, "story__title-link").text
    post_id = link[link.rfind("_") + 1:]
    tags = post.find_element(By.CLASS_NAME, "story__tags.tags").text
    post_time = post.find_element(By.TAG_NAME, "time").get_attribute("datetime")
    count_comments = post.find_element(By.CLASS_NAME,
                                       "story__comments-link-count").text
    rating = swipe_side.find_element(By.CLASS_NAME, "story__rating-count").text
    data.append([post_id, title, nick,
                 views, rating, count_comments,
                 tags, link, post_time])
    return link


def get_data_frame_of_site(sleep_secs, date_range):
    """
    Parse all posts in range of passed weeks. For every week function do site search in
    https://pikabu.ru/search by date.
    returns the table of site's data
    """
    added_posts = set()

    for start_date, end_date in date_range:
        try:
            print(start_date, end_date, end=" ")
            # load page with post, which was posted in range start and end dates
            load_page_by_date(start_date, end_date)
            while True:
                posts = driver.find_elements(By.XPATH, "//div[@class='story__main']")
                swipe_sides = driver.find_elements(By.XPATH, "//div[@class='story__left ']")
                displayed_posts = [
                    p.find_element(By.CLASS_NAME, "story__title-link").get_attribute("href")
                    not in added_posts for p in posts]
                if True not in displayed_posts:  # everything we already saw
                    print(len(added_posts), "We reach end of the page", time.asctime())
                    break
                start_ind = displayed_posts.index(True)
                posts = posts[start_ind:]  # start with post, which we didn't see
                swipe_sides = swipe_sides[start_ind:]
                for post, swipe_side in zip(posts, swipe_sides):
                    link = parse_post(post, swipe_side)
                    added_posts.add(link)
                    wait(sleep_secs[0], sleep_secs[1])
                    swipe_side.click()
                    wait(0.1, 0.5)  # just little wait in case of loading something
        except:
            print("Fuck! Something goes wrong")
            driver.save_screenshot(f"D:\\{time.asctime()}_{len(data)}.png")

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
    df.iloc[:, -1] = pd.to_datetime(df.iloc[:, -1])
    df.iloc[:, -1] = df.iloc[:, -1].dt.date
    print(df)
    # df_ = df.iloc[:, [0, 3]]
    grouped_df = df.groupby("post_time").count()
    sns.relplot(data=grouped_df, x="post_time", y="title", kind="line")
    plt.show()


def split_on_weeks(start_day, end_day, split_range=7):
    """
    :return: [(start_day, start_day + split_range), ..., (end_day - split_range, end_day)]
    :param start_day looks like "01/01/22"
    :param end_day looks like "01/02/22"
    :param split_range is how long will be our "weeks"
    """
    if start_day == end_day:
        return [[start_day, end_day]]
    if split_range <= 1:
        split_range = 2
    result = []
    frmt = "%d/%m/%y"
    s_day, e_day = datetime.datetime.strptime(start_day, frmt), \
                   datetime.datetime.strptime(end_day, frmt)
    cur_day = datetime.datetime.strptime(start_day, frmt)
    while cur_day < e_day:
        week = [cur_day, cur_day + datetime.timedelta(split_range - 1)]
        week[0], week[1] = week[0].strftime(frmt), week[1].strftime(frmt)
        result.append(week)
        cur_day += datetime.timedelta(split_range)
    result[-1][1] = end_day
    return result


if __name__ == "__main__":
    ser = Service(r"C:\Program Files (x86)\Google\chromedriver.exe")
    op = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=ser, options=op)
    driver.set_window_position(-1200, 0)
    driver.maximize_window()
    page_link = f'https://pikabu.ru/tag/Собака'
    driver.get(page_link)
    # Sign in
    sign_in = driver.find_elements(By.TAG_NAME, "form")[1]  # this is reference to sign in form
    login, password = sign_in.find_elements(By.CLASS_NAME, "input__input")
    login.send_keys("Nickname")
    time.sleep(1)
    password.send_keys("password")  # TODO change password :)
    time.sleep(3)
    driver.find_element(By.CLASS_NAME, "button_success.button_width_100").click()  # click submit
    time.sleep(15)  # maybe need insert captcha

    dates = split_on_weeks("01/01/20", "02/02/22", 10)
    data = []
    df = get_data_frame_of_site((0.75, 1.75), dates)
    df.columns = ["post_id", "title", "nick",
                  "views", "rating", "count_comments",
                  "tags", "link", "post_time"]
    driver.quit()
    df.replace(";", ".", inplace=True)
    df.to_csv(r"C:\Users\Norma\PycharmProjects\Machine "
              r"Learning\Datasets\pikabu\dog 01.20_12.20.csv",
              sep=";", encoding="UTF-8")
