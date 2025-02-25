// frontend/script.js

const backendBaseUrl = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const subtopicInput = document.getElementById("subtopicInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

const quizSubtopicInput = document.getElementById("quizSubtopicInput");
const numQuestionsInput = document.getElementById("numQuestions");
const generateBtn = document.getElementById("generateBtn");
const generateStatus = document.getElementById("generateStatus");

const fetchSubtopicInput = document.getElementById("fetchSubtopicInput");
const fetchBtn = document.getElementById("fetchBtn");
const quizList = document.getElementById("quizList");

// Upload PDF file
uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  const subtopic = subtopicInput.value.trim();

  if (!file) {
    alert("Please select a PDF file.");
    return;
  }
  if (!subtopic) {
    alert("Please enter a subtopic.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("subtopic", subtopic);

  uploadStatus.textContent = "Uploading...";

  try {
    const response = await fetch(`${backendBaseUrl}/upload`, {
      method: "POST",
      body: formData
    });
    const result = await response.json();
    uploadStatus.textContent = response.ok ? `Success: ${result.message}` : `Error: ${JSON.stringify(result)}`;
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = "Upload failed. See console for details.";
  }
});

// Generate quizzes
generateBtn.addEventListener("click", async () => {
  const subtopic = quizSubtopicInput.value.trim();
  const numQuestions = parseInt(numQuestionsInput.value, 10);
  if (!subtopic) {
    alert("Please enter a subtopic for quiz generation.");
    return;
  }

  generateStatus.textContent = "Generating quizzes...";

  try {
    const response = await fetch(`${backendBaseUrl}/generate-quizzes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subtopic, num_questions: numQuestions })
    });
    const result = await response.json();
    generateStatus.textContent = response.ok ? "Quizzes generated successfully!" : `Error: ${JSON.stringify(result)}`;
    console.log("Generated Quizzes:", result.quizzes);
  } catch (err) {
    console.error(err);
    generateStatus.textContent = "Generation failed. See console for details.";
  }
});

// Fetch quizzes for a subtopic
fetchBtn.addEventListener("click", async () => {
  const subtopic = fetchSubtopicInput.value.trim();
  if (!subtopic) {
    alert("Please enter a subtopic to fetch quizzes.");
    return;
  }
  quizList.innerHTML = "Loading...";

  try {
    const response = await fetch(`${backendBaseUrl}/quizzes?subtopic=${subtopic}`);
    const result = await response.json();
    if (response.ok) {
      quizList.innerHTML = "";
      if (result.quizzes.length === 0) {
        quizList.innerHTML = "<p>No quizzes found for this subtopic.</p>";
      } else {
        result.quizzes.forEach(q => {
          const div = document.createElement("div");
          div.classList.add("quiz-item");
          div.innerHTML = `
            <strong>Question:</strong> ${q.question}<br>
            <strong>Correct Answer:</strong> ${q.correct_answer}<br>
            <strong>Distractors:</strong> ${q.distractors.join(", ")}
          `;
          quizList.appendChild(div);
        });
      }
    } else {
      quizList.innerHTML = `Error: ${JSON.stringify(result)}`;
    }
  } catch (err) {
    console.error(err);
    quizList.innerHTML = "Fetch failed. See console for details.";
  }
});
