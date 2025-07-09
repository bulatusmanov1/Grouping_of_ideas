import flet as ft

class WorkPage:
    def __init__(self, page):
        self.page = page
        self.page.controls.clear()
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER  # Ensure horizontal alignment is also centered
        self.create_page()

    def create_page(self):
        self.page.add(ft.Column(
            [
                ft.Text("Работа с модулями", style="heading", color="#E10000"),
                ft.TextField(label="Заголовок", width=300),
                ft.TextField(label="Название", width=300),
                ft.ElevatedButton("Проверить идею", on_click=self.check_idea, width=200, color="#FFFFFF", bgcolor="#E10000")
            ],
            alignment=ft.MainAxisAlignment.CENTER
        ))

    def check_idea(self, e):
        print("Идея проверена")