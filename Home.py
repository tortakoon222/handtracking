import streamlit as st

def home_page():
    st.set_page_config(page_title="Hand Gesture Volume Control")

    st.title("Hand Gesture Volume Control")

    st.header("วัตถุประสงค์เพื่อศึกษา")
    st.write("  ในปัจจุบันมีผู้ที่ป่วยทางด้านร่างกายตั้งแต่ท่อนเอวลงไปไม่สามารถเคลื่อนที่ได้เวลารับชมสื่อต่าง ๆ ทางโทรทัศน์ เวลาที่จะปรับเสียงไม่สามรถลุกไปหยิบรีโมทมากดคำสั่งได้ ทางผู้จัดทำได้ใช้คำสั่งมือในการสั่งงานหรือการทำงาน ทางผู้จัดทำได้เล็งเห็นถึงปัญหาที่เกิดขึ้นนี้ และ ได้ทำโครงการนี้ขึ้นมา โดยนำเทคนิค Object Detection มาประยุกต์ใช้ในการสร้างระบบตรวจับการขยับมือเพิ่ม-ลดเสียง เพื่อทำการตรวจจับ สั่งงานโดยการใช้มือควบคุมแทนการใช้รีโมทคอนโทน")

    st.header("จัดทำโดย")

home_page()

