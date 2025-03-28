-- Find positions for a specific wallet address
SELECT p.*
FROM positions p
JOIN users u ON p."userId" = u.id
WHERE u."walletAddress" = '0xf2873F92324E8EC98a82C47AFA0e728Bd8E41665';