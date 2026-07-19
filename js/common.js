fetch("../components/sidebar.html")
    .then(response => response.text())
    .then(data => {
        document.getElementById("sidebar").innerHTML = data;
    })
    .catch(error => {
        console.error("サイドバーの読み込みに失敗しました:", error);
    });