import streamlit as st
import pandas as pd

df = pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    })

def main():
    st.title('데이터프레임 예시')
    st.dataframe(df)

if __name__ == '__main__':
    main()