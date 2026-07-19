const settingBtn = document.getElementById("settingBtn");
const closeBtn = document.getElementById("closeBtn");
const settingModal = document.getElementById("settingModal");

settingBtn.addEventListener("click", () => {
    settingModal.style.display = "flex";
});

closeBtn.addEventListener("click", () => {
    settingModal.style.display = "none";
});

window.addEventListener("click", (e) => {
    if (e.target === settingModal) {
        settingModal.style.display = "none";
    }
});

const refreshBtn = document.getElementById("refreshBtn");

if(refreshBtn){

    refreshBtn.addEventListener("click",()=>{

        location.reload();

    });

}


const detailModal = document.getElementById("detailModal");

const detailButtons = document.querySelectorAll(".detail-btn");

const closeDetailBtn = document.getElementById("closeDetailBtn");

detailButtons.forEach(button => {

    button.addEventListener("click", () => {

        document.getElementById("detailUser").textContent =
            button.dataset.user;

        document.getElementById("detailType").textContent =
            button.dataset.type;

        document.getElementById("detailTime").textContent =
            button.dataset.time;

        document.getElementById("detailStatus").textContent =
            button.dataset.status;

        document.getElementById("detailMessage").value =
            button.dataset.message;

        detailModal.style.display = "flex";

    });

});

closeDetailBtn.addEventListener("click", () => {

    detailModal.style.display = "none";

});

const searchInput = document.getElementById("searchInput");

if(searchInput){

    searchInput.addEventListener("keyup",()=>{

        const keyword = searchInput.value.toLowerCase();

        const rows = document.querySelectorAll("#notificationTable tr");

        rows.forEach((row,index)=>{

            if(index===0) return;

            const text = row.innerText.toLowerCase();

            row.style.display =
                text.includes(keyword) ? "" : "none";

        });

    });

}

const sortSelect = document.getElementById("sortSelect");

if(sortSelect){

    sortSelect.addEventListener("change",()=>{

        const tbody = document.querySelector("#notificationTable tbody");

        const rows = Array.from(tbody.querySelectorAll("tr"));

        if(sortSelect.value==="new"){

            rows.reverse();

        }else{

            rows.reverse();

        }

        tbody.innerHTML="";

        rows.forEach(row=>tbody.appendChild(row));

    });

}

const rows = document.querySelectorAll("#notificationTable tbody tr");

const rowsPerPage = 2;

let currentPage = 1;

function showPage(page){

    rows.forEach((row,index)=>{

        row.style.display="none";

        if(index >= (page-1)*rowsPerPage &&
           index < page*rowsPerPage){

            row.style.display="";

        }

    });

    pageNumber.textContent = page;

}

showPage(currentPage);

nextPage.addEventListener("click",()=>{

    if(currentPage*rowsPerPage < rows.length){

        currentPage++;

        showPage(currentPage);

    }

});

prevPage.addEventListener("click",()=>{

    if(currentPage>1){

        currentPage--;

        showPage(currentPage);

    }

});