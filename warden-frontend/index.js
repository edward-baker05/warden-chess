

// Initialize the chess.js game
const game = new Chess();

// Get the board element
const board = document.getElementById("boardInner");

// Define a function that renders the board based on the FEN position
const renderBoardFromFen = (fen) => {
    // Split the FEN position into its individual parts
    const parts = fen.split(" ");
    // The first part is the piece placement
    const placement = parts[0];
    // Split the placement into rows
    const rows = placement.split("/");
    // Clear the board
    board.innerHTML = "";
    // Loop through the rows and create the squares
    rows.forEach((row, i) => {
        // Create a row element
        const rowEl = document.createElement("div");
        rowEl.className = "row";
        let colIndex = 0;
        // Loop through the characters in the row
        for (const c of row) {
            // Check if the character is a number (indicating an empty square)
            if (!isNaN(c)) {
                // Create the empty squares
                for (let j = 0; j < c; j++) {
                    colIndex++;
                    const square = document.createElement("div");
                    square.className = "square";
                    // Set the background colour of the square based on its position in the row
                    square.style.backgroundColor =
                        colIndex % 2 === 0 ? (i % 2 === 0 ? "#eeeed2" : "#769656") : (i % 2 === 0 ? "#769656" : "#eeeed2");
                    rowEl.appendChild(square);
                }
            } else {
                // The character is a piece
                colIndex++;
                // Determine the colour and type of the piece
                const colour = c.toLowerCase() === c ? "#769656" : "#eeeed2";
                const type = c.toLowerCase();
                // Create a square element for the piece
                const square = document.createElement("div");
                square.className = "square";
                // Set the background colour of the square based on its position in the row
                square.style.backgroundColor =
                    colIndex % 2 === 0 ? (i % 2 === 0 ? "#eeeed2" : "#769656") : (i % 2 === 0 ? "#769656" : "#eeeed2");
                // Create an img element for the piece
                const img = document.createElement("img");
                const filename = colour === '#769656' ? `black${type}` : `white${type}`;
                img.src = `./assets/${filename}.svg`;
                img.className = "piece";
                // Add the img element to the square
                square.appendChild(img);
                rowEl.appendChild(square);
            }
        }
        // Add the row element to the board
        board.appendChild(rowEl);
        // Enable drag-and-drop for the pieces
        const pieces = document.querySelectorAll(".piece");
        pieces.forEach((piece) => {
            piece.addEventListener("dragstart", (event) => {
                // Store the square element that the piece belongs to in the dataTransfer object
                event.dataTransfer.setData("square", event.target.parentElement.id);
            });
        });

        // Enable drop behavior for the squares
        const squares = document.querySelectorAll(".square");
        squares.forEach((square) => {
            square.addEventListener("dragover", (event) => {
                // Allow dropping onto the square
                event.preventDefault();
            });
            square.addEventListener("drop", (event) => {
                // Get the square element that the piece belongs to from the dataTransfer object
                const sourceSquare = document.getElementById(event.dataTransfer.getData("square"));
                // Check if the move is valid
                const move = checkMove(sourceSquare, square);
                if (move) {
                    // Update the game state
                    game.move(move);
                    // Render the board based on the new game state
                    renderBoardFromFen(game.fen());
                }
            });
        });

const checkMove = (source, target) => {
    // Get the source square's rank and file
    const sourceRank = source.parentElement.getAttribute("data-rank");
    const sourceFile = source.getAttribute("data-file");
    // Get the target square's rank and file
    const targetRank = target.parentElement.getAttribute("data-rank");
    const targetFile = target.getAttribute("data-file");
    // Create the move object
    const move = {
        from: `${sourceFile}${sourceRank}`,
        to: `${targetFile}${targetRank}`,
    };
    // Check if the move is valid
    return game.move(move);
        };
    });
};

// Render the initial position of the chess game
renderBoardFromFen(game.fen());
