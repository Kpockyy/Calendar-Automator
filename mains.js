let rowCount = 1;

// Drag and drop functionality for ranking
const rankingContainer = document.getElementById('rankingContainer');
let draggedItem = null;

rankingContainer.addEventListener('dragstart', (e) => {
    if (e.target.classList.contains('ranking-item')) {
        draggedItem = e.target;
        e.target.classList.add('dragging');
    }
});

rankingContainer.addEventListener('dragend', (e) => {
    if (e.target.classList.contains('ranking-item')) {
        e.target.classList.remove('dragging');
        updateRankNumbers();
    }
});

rankingContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    const afterElement = getDragAfterElement(rankingContainer, e.clientY);
    if (afterElement == null) {
        rankingContainer.appendChild(draggedItem);
    } else {
        rankingContainer.insertBefore(draggedItem, afterElement);
    }
});

function getDragAfterElement(container, y) {
    const draggableElements = [...container.querySelectorAll('.ranking-item:not(.dragging)')];
    
    return draggableElements.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        
        if (offset < 0 && offset > closest.offset) {
            return { offset: offset, element: child };
        } else {
            return closest;
        }
    }, { offset: Number.NEGATIVE_INFINITY }).element;
}

function updateRankNumbers() {
    // Rank numbers removed - no longer needed
}

// Course table functions
function addRow() {
    rowCount++;
    const tableBody = document.getElementById('courseTableBody');
    const newRow = document.createElement('tr');
    
    newRow.innerHTML = `
        <td>${rowCount}</td>
        <td>
            <select>
                <option value="" selected></option>
                <option value="assignment">Assignment</option>
                <option value="writing">Writing</option>
                <option value="lab">Lab</option>
                <option value="project">Project</option>
                <option value="others">Others</option>
            </select>
        </td>
        <td>
            <select>
                <option value="" selected></option>
                <option value="business">Business</option>
                <option value="stem">STEM</option>
                <option value="humanities">Humanities</option>
                <option value="arts">Arts</option>
                <option value="others">Others</option>
            </select>
        </td>
        <td>
            <input type="text" placeholder="">
        </td>
        <td>
            <input type="text" placeholder="">
        </td>
        <td>
            <select>
                <option value="" selected></option>
                <option value="1-2">1-2</option>
                <option value="3-5">3-5</option>
                <option value="6-10">6-10</option>
                <option value="10+">10+</option>
            </select>
        </td>
        <td>
            <select>
                <option value="" selected></option>
                <option value="textbook">Textbook</option>
                <option value="google">Google</option>
                <option value="ai">AI</option>
                <option value="mixed">Mixed</option>
            </select>
        </td>
        <td>
            <button class="delete-btn" onclick="deleteRow(this)">Delete</button>
        </td>
    `;
    
    tableBody.appendChild(newRow);
}

function deleteRow(button) {
    const row = button.parentElement.parentElement;
    row.remove();
    updateRowNumbers();
}

function updateRowNumbers() {
    const rows = document.querySelectorAll('#courseTableBody tr');
    rowCount = rows.length;
    rows.forEach((row, index) => {
        row.cells[0].textContent = index + 1;
    });
}