// var PythonShell = require('python-shell').PythonShell

var board = null
var game = new Chess()
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

    // If the move is a pawn promotion, create a selection box with 4 options for promotion
    if (move.flags.includes('p')) {
        var promotion = ''
        var choices = ['Queen', 'Rook', 'Bishop', 'Knight']

        // Create the selection box
        var container = document.createElement('div')
        container.classList.add('promotion-box')
        for (var i = 0; i < choices.length; i++) {
            var choice = document.createElement('div')
            choice.classList.add('promotion-choice')
            choice.innerHTML = choices[i]
            container.appendChild(choice)
        }
        document.body.appendChild(container)

        // Add event listeners to the promotion choices
        var promotionChoices = document.querySelectorAll('.promotion-choice')
        promotionChoices.forEach(function (choice) {
            choice.addEventListener('click', function () {
                // Remove the selection box
                document.body.removeChild(container)

                // Set the promotion based on the chosen option
                switch (choice.innerHTML) {
                    case 'Queen':
                        promotion = 'q'
                        break
                    case 'Rook':
                        promotion = 'r'
                        break
                    case 'Bishop':
                        promotion = 'b'
                        break
                    case 'Knight':
                        promotion = 'n'
                        break
                }

                // Make the move with the chosen promotion
                move = game.move({
                    from: source,
                    to: target,
                    promotion: promotion
                })
            })
        })
    }

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
        onSnapEnd: onSnapEnd
    }

    board = Chessboard('myBoard', config)
}

function getColour() {
    // Add a dropdown menu for the player to select their colour
    var colourSelect = document.createElement('select')
    colourSelect.id = 'player-colour'
    var blankOption = document.createElement('option')
    blankOption.innerHTML = ''
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
            }
        } else {
            selectedColour = document.getElementById('player-colour').value
            if (selectedColour === 'White') {
                playerColour = 'w'
            } else if (selectedColour === 'Black') {
                playerColour = 'b'
            }
        }
    })
}

// JavaScript code
async function sendStringToPython(body) {
    try {
        const resp = await fetch('/move.py', { method: 'POST', body })
        return await resp.text()
    } catch (data) {
        return console.error(data)
    }
}

// If start button pressed, send string to python
document.getElementById('startBtn').addEventListener('click', async () => {
    const resp = await sendStringToPython('Hello World!')
    console.log(resp)
})

getColour()