# merge feature-chess-improvements into feature
cd ~/dev/alphazero
git merge feature-chess-improvements

# remove obsolete worktree
git worktree remove ../alphazero-merge
