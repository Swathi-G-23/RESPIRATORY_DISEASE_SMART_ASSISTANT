with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

target1 = """    if uploaded_video is not None:
        with st.container():
            st.markdown("<div class='video-panel'>", unsafe_allow_html=True)
            st.video(uploaded_video)
            st.markdown("</div>", unsafe_allow_html=True)"""

replacement1 = """    if uploaded_video is not None:
        st.markdown("<div class='media-container'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div class='video-panel'>", unsafe_allow_html=True)
            st.video(uploaded_video)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)"""

code = code.replace(target1, replacement1)

target2 = """                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB", use_column_width=True)"""

replacement2 = """                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                st.markdown("<div class='media-container'>", unsafe_allow_html=True)
                col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
                with col_img2:
                    stframe.image(frame, channels="RGB", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)"""

code = code.replace(target2, replacement2)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
