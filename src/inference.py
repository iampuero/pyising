
def inference(loadfolder, savefolder, loadfile, etaH, etaJ):
    # Load data
    data = scipy.io.loadmat(os.path.join(loadfolder, loadfile))
    binnedSpikes = data['binnedSpikes']
    binnedSpikes = np.asarray(binnedSpikes, dtype=bool)
    N, T = binnedSpikes.shape

    # MEAN AND COVARIANCE OF SUFFICIENT STATISTICS
    offMat = np.zeros((N, N), dtype=int)
    for n1 in range(N):
        offMat[n1, n1] = n1 + 1
        for n2 in range(n1 + 1, N):
            off_value = offset(n1 + 1, N) + n2 + 1
            offMat[n1, n2] = off_value
            offMat[n2, n1] = off_value

    chi = np.zeros((D, D))
    for b in range(T):
        L = np.nonzero(binnedSpikes[:, b])[0]
        index = offMat[L.reshape(-1, 1), L]
        index_flat = index.flatten()
        chi[np.ix_(index_flat, index_flat)] += 1

    p = np.diag(chi) / T
    chiC = chi / T - np.outer(p, p)
    eV = np.sort(eigvalsh(chiC))

    # INITIAL CONDITION
    global params

    jListIn = np.zeros(N * (N - 1) // 2)
    hListIn = np.log(p[:N] / (1 - p[:N]))
    params = np.hstack((hListIn, jListIn))

    # INFERENCE REGULARIZATION
    etaMat = np.diag(np.hstack((etaH * np.ones(N), etaJ * np.ones(N * (N - 1) // 2))))

    # PARAMETERS INFERENCE
    num_workers = None  # Use the default number of workers

    logTd = 3
    nStepMore = 0  # Increase this for posterior sampling
    threshold = np.exp(-0.0)
    alphaStart = 1.0
    verbose = True

    # Make sure the dataDriven_IsingModel function is defined before calling it
    q, output = dataDriven_IsingModel(N, T, 1 / eV[0], logTd, p, chiC, etaMat, alphaStart, threshold, nStepMore, num_workers, verbose)

    # PLOTTING CORRELATIONS
    ccData = np.zeros(N * (N - 1) // 2)
    ccModel = np.zeros(N * (N - 1) // 2)

    for n1 in range(N):
        for n2 in range(n1 + 1, N):
            ccData[offset(n1 + 1, N) + n2 - N] = p[offset(n1 + 1, N) + n2] - p[n1] * p[n2]
            ccModel[offset(n1 + 1, N) + n2 - N] = q[offset(n1 + 1, N) + n2] - q[n1] * q[n2]

    # Save results
    scipy.io.savemat(os.path.join(savefolder, loadfile), {
        'hListIn': hListIn,
        'jListIn': jListIn,
        'p': p,
        'q': q,
        'output': output,
        'params': params,
        'ccData': ccData,
        'ccModel': ccModel
        })

