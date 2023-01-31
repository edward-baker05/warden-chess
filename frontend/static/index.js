var board = null
var game = new Chess()
game.load("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")
var whiteSquareGrey = '#a9a9a9'
var blackSquareGrey = '#696969'
var playerColour = null

function removeGreySquares() {
    $('#myBoard .square-55d63').css('background', '')
}

function greySquare(square) {
    var $square = $('#myBoard .square-' + square)

    var background = whiteSquareGrey
    if ($square.hasClass('black-3c85d')) {
        background = blackSquareGrey
    }

    $square.css('background', background)
}

function onDragStart(source, piece) {
    // do not pick up pieces if the game is over
    if (game.game_over()) return false

    // or if it's not that side's turn
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return 'snapback'
    }
}

function onDrop(source, target) {
    removeGreySquares()

    console.log(typeof(source));
    console.log(typeof(target));

    // check if it is the player's turn
    if (game.turn() !== playerColour) {
        return 'snapback'
    }

    // Check if the move is a pawn promotion
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    })

    // illegal move
    if (move === null) return 'snapback'

    // check if the game is over
    if (game.game_over()) {
        // Create a popup saying the game is over
        var popup = document.createElement('div')
        popup.classList.add('popup')
        popup.innerHTML = 'Game Over'
        document.body.appendChild(popup)
    }

    getAIMove();
}

function aiMove(source, target) {
    removeGreySquares()

    // Make the AI move
    game.move({
        from: source,
        to: target,
        promotion: 'q'
    })

    // check if the game is over
    if (game.game_over()) {
        // Create a popup saying the game is over
        var popup = document.createElement('div')
        popup.classList.add('popup')
        popup.innerHTML = 'Game Over'
        document.body.appendChild(popup)
    }

    onSnapEnd();

    // getAIMove();
}

function onMouseoverSquare(square, piece) {
    // get list of possible moves for this square
    var moves = game.moves({
        square: square,
        verbose: true
    })

    // exit if there are no moves available for this square
    if (moves.length === 0) return

    // highlight the square they moused over
    greySquare(square)

    // highlight the possible squares for this piece
    for (var i = 0; i < moves.length; i++) {
        greySquare(moves[i].to)
    }
}

function onMouseoutSquare(square, piece) {
    removeGreySquares()
}

function onSnapEnd() {
    board.position(game.fen())
    // Check if the game is in checkmate
    if (game.in_checkmate() === true) {
        var king_color = game.turn() === 'w' ? 'b' : 'w'
        var kingSquare = game.king_position(king_color)
        var kingElement = document.querySelector('.square-' + kingSquare)
        kingElement.style.backgroundColor = 'red'
    }
}

function chooseColour(colour) {
    playerColour = colour
    var config = {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onMouseoutSquare: onMouseoutSquare,
        onMouseoverSquare: onMouseoverSquare,
        onSnapEnd: onSnapEnd,
        position: game.fen()
    }

    board = Chessboard('myBoard', config)
}

function getColour() {
    // Add a dropdown menu for the player to select their colour
    var colourSelect = document.createElement('select')
    colourSelect.id = 'player-colour'
    var blankOption = document.createElement('option')
    blankOption.innerHTML = 'Click here to choose your colour'
    colourSelect.appendChild(blankOption)
    var whiteOption = document.createElement('option')
    whiteOption.innerHTML = 'White'
    var blackOption = document.createElement('option')
    blackOption.innerHTML = 'Black'
    colourSelect.appendChild(whiteOption)
    colourSelect.appendChild(blackOption)
    document.body.appendChild(colourSelect)

    colourSelect.addEventListener('change', function () {
        var selectedColour = document.getElementById('player-colour').value
        if (playerColour !== 'w' && playerColour !== 'b') {
            if (selectedColour === 'White') {
                chooseColour('w')
            } else if (selectedColour === 'Black') {
                chooseColour('b')
                getAIMove();
            }
        } else {
            selectedColour = document.getElementById('player-colour').value
            if (selectedColour === 'White') {
                playerColour = 'w'
            } else if (selectedColour === 'Black') {
                playerColour = 'b'
                getAIMove();
            }
        }
    })

    select.remove(0);
}

function getAIMove() {
    console.log("Quering AI for next move...");
    const data = game.fen();
    $.getJSON('/get_move', { fen: data },
        function (response) {
            const fen_result = response.result;
            aiMove(fen_result[0], fen_result[1]);
        });
    return true;
}

getColour()