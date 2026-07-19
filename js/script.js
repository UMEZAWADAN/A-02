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