import nltk
import streamlit as st
import preprocessor
import helper
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# App title
st.sidebar.title("Whatsapp Chat  Sentiment Analyzer")

# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
nltk.downloader.download('vader_lexicon')

# File upload button
uploaded_file = st.sidebar.file_uploader("Choose a file")

# Main heading
st.markdown("<h1 style='text-align: center; color: red;'>Whatsapp Chat  Sentiment Analyzer</h1>",
            unsafe_allow_html=True)

if uploaded_file is not None:
    # Getting byte form & then decoding
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    # Perform preprocessing
    df = preprocessor.preprocess(data)
    st.dataframe(df)

    # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Object
    sentiments = SentimentIntensityAnalyzer()

    # Creating different columns for (Positive/Negative/Neutral)
    df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
    df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
    df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]  # Neutral


    # To indentify true sentiment per row in message column
    def sentiment(data):
        if data["po"] >= data["ne"] and data["po"] >= data["nu"]:
            return 1
        if data["ne"] >= data["po"] and data["ne"] >= data["nu"]:
            return -1
        if data["nu"] >= data["po"] and data["nu"] >= data["ne"]:
            return 0


    # Creating new column & Applying function
    df['value'] = df.apply(lambda row: sentiment(row), axis=1)

    # User names list
    user_list = df['user'].unique().tolist()

    # Sorting
    user_list.sort()

    # Insert "Overall" at index 0
    user_list.insert(0, "Overall")

    # Selectbox
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Positive)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map(selected_user, df, 1)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map(selected_user, df, 0)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Negative)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map(selected_user, df, -1)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # weekly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: blue;'>Daily Activity map(Positive)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map(selected_user, df, 1)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: blue;'>Daily Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map(selected_user, df, 0)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: blue;'>Daily Activity map(Negative)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map(selected_user, df, -1)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.markdown("<h3 style='text-align: center; color: yellow;'>Weekly Activity Map(Positive)</h3>",
                            unsafe_allow_html=True)

                user_heatmap = helper.activity_heatmap(selected_user, df, 1)

                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: yellow;'>Weekly Activity Map(Neutral)</h3>",
                            unsafe_allow_html=True)

                user_heatmap = helper.activity_heatmap(selected_user, df, 0)

                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: yellow;'>Weekly Activity Map(Negative)</h3>",
                            unsafe_allow_html=True)

                user_heatmap = helper.activity_heatmap(selected_user, df, -1)

                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')

        # Daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: orange;'>Daily Timeline(Positive)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline(selected_user, df, 1)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: orange;'>Daily Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline(selected_user, df, 0)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: orange;'>Daily Timeline(Negative)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline(selected_user, df, -1)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: green;'>Monthly Timeline(Positive)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline(selected_user, df, 1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: green;'>Monthly Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline(selected_user, df, 0)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: green;'>Monthly Timeline(Negative)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline(selected_user, df, -1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Percentage contributed
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: orange;'>Most Positive Contribution</h3>",
                            unsafe_allow_html=True)
                x = helper.percentage(df, 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: orange;'>Most Neutral Contribution</h3>",
                            unsafe_allow_html=True)
                y = helper.percentage(df, 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: orange;'>Most Negative Contribution</h3>",
                            unsafe_allow_html=True)
                z = helper.percentage(df, -1)

                # Displaying
                st.dataframe(z)

        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            # Getting names per sentiment
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1, col2, col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: cyan;'>Most Positive Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: cyan;'>Most Neutral Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: cyan;'>Most Negative Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # Most common positive words
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # Data frame of most common positive words.
                most_common_df = helper.most_common_words(selected_user, df, 1)

                # heading
                st.markdown("<h3 style='text-align: center; color: magenta;'>Positive Words</h3>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Display error image
                st.image('error.webp')
        with col2:
            try:
                # Data frame of most common neutral words.
                most_common_df = helper.most_common_words(selected_user, df, 0)

                # heading
                st.markdown("<h3 style='text-align: center; color: magenta;'>Neutral Words</h3>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Display error image
                st.image('error.webp')
        with col3:
            try:
                # Data frame of most common negative words.
                most_common_df = helper.most_common_words(selected_user, df, -1)

                # heading
                st.markdown("<h3 style='text-align: center; color: magenta;'>Negative Words</h3>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Display error image
                st.image('error.webp')

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)
