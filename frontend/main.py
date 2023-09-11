import streamlit as st 
import pandas as pd
import requests
import datetime
import altair as alt
from datetime import datetime
# logged_in = requests.get("http://localhost:8000/api/users/me")
# if logged_in.status_code == 200:
#     logged_in = True
# if logged_in.status_code == 401:
#     logged_in = False
logged_in = True
def preprocess_data(raw_data):
    data = raw_data
    data["timestamp"] = pd.to_datetime(data["created_date"])
    data["date"] = data["timestamp"].dt.date
    data["year_month"] = data["timestamp"].dt.strftime("%Y-%m")
    data["year_week"] = data["timestamp"].dt.strftime("%Y-%W")
    data["month_of_year"] = data["timestamp"].dt.month
    data["week_of_year"] = data["timestamp"].dt.strftime('%U')
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["day_name"] = data["timestamp"].dt.day_name()
    data["diff"] = data.groupby("year_month")["calo_diff"].diff().fillna(0)
    return data

@st.cache_data
def load_data():
    response = requests.get("http://localhost:8000/api/calories")
    raw_data = pd.DataFrame(response.json())
    dict = response.json()
    df = pd.json_normalize(dict['calories'])
    df = df.rename(columns={
        'calories.created_date': 'created_date',
        'calories.id': 'id',
        'calories.calo_in': 'calo_in',
        'calories.calo_out': 'calo_out',
        'calories.calo_diff': 'calo_diff'
    })
    data = preprocess_data(df)
    return data

@st.cache_resource
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

#Main page
st.title("Calories Manager App")
st.write("Calories tracking and analysis")
st.sidebar.title("User Action")
#Login and logout
if not logged_in:
    st.sidebar.subheader("Login")
    with st.sidebar:
        with st.form(key = "login_form", clear_on_submit=True):
            email = st.text_input("Email:")
            password = st.text_input("Password:", type="password", ) 
            loggin_button = st.form_submit_button("Login")
    if loggin_button:
        logged= requests.post("http://localhost:8000/api/auth/login", json={"email": email, "password": password})
        if logged.status_code == 400:
            st.sidebar.error("Wrong user name or password")
        if logged.status_code == 422:
            st.sidebar.error("Invalid user name or password")
        if logged.status_code == 200:
            logged_in = True
            st.sidebar.success("Logged in")
if not logged_in:
    st.write("Please login to view data!")
if logged_in:

    #Add data
    st.sidebar.subheader("Add Record")
    with st.sidebar:
        with st.form(key = "Add"):
            created_date = st.date_input("Measured at:")
            created_date = datetime(created_date.year, created_date.month, created_date.day)
            calo_in = st.number_input("Calories in")
            calo_out = st.number_input("Calories out")
            calo_diff = calo_in - calo_out
            add_button = st.form_submit_button("Add")
    if add_button:
        added= requests.post("http://localhost:8000/api/calories", json={"created_date": created_date.strftime("%m/%d/%Y"), "calo_in": calo_in, "calo_out": calo_out, "calo_diff": calo_diff})
        if added.status_code == 422:
            st.sidebar.error("Invalid values")
        if added.status_code == 409:
            st.sidebar.error("Record already exists")
        if added.status_code == 201:
            st.sidebar.success("Added")

    #Delete data
    st.sidebar.subheader("Delete Record")
    with st.sidebar:
        with st.form(key = "Delete"):
            delete_date = st.date_input("Delete day:")
            delete_date = datetime(delete_date.year, delete_date.month, delete_date.day)
            delete_button = st.form_submit_button("Delete")
    if delete_button:
        deleted= requests.delete("http://localhost:8000/api/calories" + "/" + delete_date.strftime("%Y-%m-%d"))
        if deleted.status_code == 422:
            st.sidebar.error("Invalid values")
        if deleted.status_code == 404:
            st.sidebar.error("Not found record on this date")
        if deleted.status_code == 204:
            st.sidebar.success("Deleted")

    #Update data
    st.sidebar.subheader("Update Record")
    with st.sidebar:
        with st.form(key = "Update"):
            update_date = st.date_input("Update day:")
            update_date = datetime(update_date.year, update_date.month, update_date.day)
            calo_in = st.number_input("Calories in:")
            calo_out = st.number_input("Calories out:")
            calo_diff = calo_in - calo_out
            update_button = st.form_submit_button("Update")
    if update_button:
        updated= requests.put("http://localhost:8000/api/calories" + "/" + update_date.strftime("%Y-%m-%d"), json={"calo_in": calo_in, "calo_out": calo_out, "calo_diff": calo_diff})
        if updated.status_code == 422:
            st.sidebar.error("Invalid values")
        if updated.status_code == 404:
            st.sidebar.error("Not found record on this date")
        if updated.status_code == 200:
            st.sidebar.success("Updated")
#Content
    data = load_data()
    csv = convert_df(data)

    #Input CSV
    st.sidebar.subheader('Input CSV')
    uploaded_file = st.sidebar.file_uploader("Choose a file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, usecols=["created_date", "calo_in", "calo_out", "calo_diff"], parse_dates=True)
        df.columns = ["created_date", "calo_in", "calo_out", "calo_diff"]
        string_columns = df.select_dtypes(include=['object']).columns
        if 'calo_in' in string_columns or 'calo_out' in string_columns or 'calo_diff' in string_columns:
            st.error("Uploadfile Format Error: Calories must be a number")
        else:
            df = df.reset_index()
            count = 0
            success = 0
            for index, row in df.iterrows():
                count = count + 1
                date = row['created_date']
                date = datetime.strptime(date, "%m/%d/%Y")
                date = datetime.strftime(date, "%m/%d/%Y")
                added= requests.post("http://localhost:8000/api/calories", json={"created_date": date, "calo_in": row['calo_in'], "calo_out": row['calo_out'], "calo_diff": row['calo_diff']})
                if added.status_code == 201:
                    success = success + 1
                if added.status_code == 422:
                    error = "Record"+ str(count) + " has Invalid values"
                    st.error(error)
                    break
                if added.status_code == 409:
                    error = "Record"+ str(count) + " already exists"
                    st.error(error)
                    break
            st.success("Added " + str(success) + " records")            
    
    #Dowload CSV
    st.sidebar.subheader("Download CSV")
    st.sidebar.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )

    d = pd.to_datetime(st.date_input("Calories track since:", data.timestamp.min()), utc=True)
    c1 = (
        alt.Chart(data[data.date > d.date()])
        .mark_line(color="blue", size=1)
        .encode(
            alt.Y(
                "calo_in",
                scale=alt.Scale(domain=[data["calo_in"].max() * 0.4, data["calo_in"].max() * 1.1]),
                title="Calories in",
            ),
            x=alt.X("date:T", title="Date"),
            tooltip=["id", "calo_in"],
        )
        .properties(height=400, width=300)
        .interactive()
    )
    c11 = (
        alt.Chart(data[data.date > d.date()])
        .mark_line(color="red", size=1)
        .encode(
            alt.Y(
                "calo_out",
                scale=alt.Scale(domain=[data["calo_out"].max() * 0.4, data["calo_out"].max() * 1.1]),
                title="Calories out",
            ),
            x=alt.X("date:T", title="Date"),
            tooltip=["id", "calo_out"],
        )
        .properties(height=400, width=300)
        .interactive()
    )
    c2 = (
        alt.Chart(data[data.date > d.date()].groupby("date")["calo_diff"].sum().reset_index())
        .mark_bar()
        .encode(
            y=alt.Y("date", title="date"),
            x=alt.X("calo_diff:Q", title="Calories remained by day"),
            color=alt.condition(
                alt.datum.calo_diff < 0,
                alt.value("green"), 
                alt.value("red"), 
            ),
            tooltip=["calo_diff"],
        )
        .properties(height=400, width=200)
        .interactive()
    )

    c3 = c1 | c11

    st.altair_chart(c3, use_container_width=True)

    st.altair_chart(c2, use_container_width=True)

    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    options = st.selectbox(
        "Calories remained by week:",
        data.year_week.unique().tolist(),
        index=data.year_week.unique().tolist().index(data.year_week.unique().tolist()[-1]),
    )

    c4 = (
        alt.Chart(data[data.year_week == options])
        .mark_circle(color="yellow")
        .encode(
            y=alt.Y(
                "calo_diff",
                scale=alt.Scale(
                    domain=[
                        -1000,
                        1000
                    ]
                ),
                title="Calories",
            ),
            x=alt.X("day_name", sort=weekdays, title="Day of week"),
            tooltip=["calo_diff", "created_at"],
        )
        .interactive()
    )

    st.altair_chart(c4, use_container_width=True)

    c5 = (
        alt.Chart(data[data.date > d.date()].groupby("date")["calo_diff"].sum().reset_index())
        .mark_bar(color="orange", size=1)
        .encode(
            alt.Y(
                "calo_diff",
                scale=alt.Scale(domain=[data["calo_diff"].min() * 1.1, data["calo_diff"].max() * 1.1]),
                title="Calories remained by date",
            ),
            x=alt.X("date:T", title="Date"),
            tooltip=["calo_diff"],
        )
        .properties(height=400, width=300)
        .interactive()
    )


    st.altair_chart(c5, use_container_width=True)

   