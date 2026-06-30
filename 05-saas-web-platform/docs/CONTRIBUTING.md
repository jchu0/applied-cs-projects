# Contributing to SaaS Web Platform

Thank you for your interest in contributing to the SaaS Web Platform! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Commit Guidelines](#commit-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We pledge to:

- Be respectful and considerate in all interactions
- Welcome contributors from all backgrounds and experience levels
- Accept constructive criticism gracefully
- Focus on what is best for the community and project
- Show empathy towards other community members

### Unacceptable Behavior

The following behaviors are considered unacceptable:

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing private information without consent
- Any conduct that could reasonably be considered inappropriate

### Reporting Issues

If you experience or witness unacceptable behavior, please report it to conduct@saas-platform.com. All reports will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Development Environment**
   - Python 3.11+
   - Node.js 18+
   - Docker and Docker Compose
   - Git

2. **Accounts**
   - GitHub account
   - Local development setup (see [DEPLOYMENT.md](./DEPLOYMENT.md))

3. **Knowledge**
   - Familiarity with Django and React/Next.js
   - Understanding of REST APIs
   - Basic knowledge of Docker and containerization

### Setting Up Your Development Environment

1. **Fork the Repository**
   ```bash
   # Fork via GitHub UI, then clone
   git clone https://github.com/YOUR-USERNAME/saas-platform.git
   cd saas-platform
   ```

2. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/original-org/saas-platform.git
   git fetch upstream
   ```

3. **Install Dependencies**
   ```bash
   # Backend
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt

   # Frontend
   cd ../frontend
   npm install
   ```

4. **Setup Pre-commit Hooks**
   ```bash
   # Install pre-commit
   pip install pre-commit

   # Setup hooks
   pre-commit install
   ```

5. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

## Development Process

### 1. Find or Create an Issue

- Check existing issues for something you'd like to work on
- If creating a new issue, provide clear description and context
- Wait for maintainer feedback before starting major work
- Comment on the issue to claim it

### 2. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or fixes
- `chore/` - Maintenance tasks

### 3. Make Your Changes

Follow these guidelines while developing:

- Write clean, readable code
- Follow existing patterns and conventions
- Add tests for new functionality
- Update documentation as needed
- Keep changes focused and atomic

### 4. Test Your Changes

```bash
# Backend tests
cd backend
pytest
pytest --cov=apps --cov-report=html

# Frontend tests
cd frontend
npm test
npm run test:coverage

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### 5. Submit a Pull Request

- Push your branch to your fork
- Create a pull request against the main repository
- Fill out the PR template completely
- Link related issues
- Wait for review and address feedback

## Code Style Guidelines

### Python/Django Style Guide

We follow PEP 8 with some modifications:

```python
# Good example
from typing import Optional, List
from django.db import models
from django.contrib.auth.models import User


class Project(models.Model):
    """
    Represents a project in the system.

    Attributes:
        name: The project name (max 255 chars)
        owner: The user who owns this project
        created_at: Timestamp of creation
    """

    name = models.CharField(max_length=255, help_text="Project name")
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="owned_projects"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner', '-created_at']),
        ]

    def __str__(self) -> str:
        return self.name

    def get_members(self) -> List[User]:
        """
        Get all members of this project.

        Returns:
            List of User objects who are members
        """
        return self.members.all()

    @classmethod
    def create_with_owner(cls, name: str, owner: User) -> 'Project':
        """
        Create a new project with the given owner.

        Args:
            name: The project name
            owner: The user who will own the project

        Returns:
            The created Project instance
        """
        project = cls.objects.create(name=name, owner=owner)
        project.members.add(owner)
        return project
```

**Key Points:**
- Use type hints for function arguments and return values
- Add docstrings for all classes and public methods
- Use meaningful variable names
- Keep lines under 88 characters (Black formatter)
- Group imports: standard library, third-party, local

### JavaScript/TypeScript Style Guide

We use ESLint and Prettier for JavaScript/TypeScript:

```typescript
// Good example
import React, { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useQuery, useMutation } from '@tanstack/react-query';

import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { projectApi } from '@/services/api/projects';
import type { Project, User } from '@/types';

interface ProjectCardProps {
  project: Project;
  onUpdate?: (project: Project) => void;
  className?: string;
}

/**
 * Display a project card with basic information and actions.
 */
export const ProjectCard: React.FC<ProjectCardProps> = ({
  project,
  onUpdate,
  className = '',
}) => {
  const router = useRouter();
  const [isEditing, setIsEditing] = useState(false);

  // Fetch project details
  const { data, isLoading } = useQuery({
    queryKey: ['project', project.id],
    queryFn: () => projectApi.getProject(project.id),
    enabled: isEditing,
  });

  // Update project mutation
  const updateMutation = useMutation({
    mutationFn: projectApi.updateProject,
    onSuccess: (updatedProject) => {
      setIsEditing(false);
      onUpdate?.(updatedProject);
    },
  });

  const handleEdit = useCallback(() => {
    setIsEditing(true);
  }, []);

  const handleSave = useCallback(
    async (updates: Partial<Project>) => {
      await updateMutation.mutateAsync({
        id: project.id,
        ...updates,
      });
    },
    [project.id, updateMutation]
  );

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      setIsEditing(false);
    };
  }, []);

  return (
    <Card className={`project-card ${className}`}>
      <Card.Header>
        <h3>{project.name}</h3>
        <Button size="sm" variant="ghost" onClick={handleEdit}>
          Edit
        </Button>
      </Card.Header>
      <Card.Body>
        <p>{project.description}</p>
        <div className="project-meta">
          <span>Created: {new Date(project.createdAt).toLocaleDateString()}</span>
          <span>Members: {project.membersCount}</span>
        </div>
      </Card.Body>
    </Card>
  );
};

// Default export for lazy loading
export default ProjectCard;
```

**Key Points:**
- Use functional components with hooks
- Proper TypeScript types and interfaces
- Destructure props
- Use meaningful component and variable names
- Keep components focused and single-purpose
- Add JSDoc comments for complex components

### CSS/Styling Guidelines

```scss
// Use CSS Modules or styled-components
.project-card {
  @apply bg-white rounded-lg shadow-sm border border-gray-200;

  &:hover {
    @apply shadow-md border-gray-300;
  }

  .project-meta {
    @apply flex justify-between text-sm text-gray-600 mt-4;

    span {
      @apply flex items-center gap-1;
    }
  }
}

// Component-specific styles
.project-card {
  --card-padding: 1rem;
  --card-radius: 0.5rem;

  padding: var(--card-padding);
  border-radius: var(--card-radius);

  @media (min-width: 768px) {
    --card-padding: 1.5rem;
  }
}
```

## Testing Guidelines

### Unit Testing

#### Backend (Python/Django)

```python
# tests/test_projects.py
import pytest
from django.test import TestCase
from django.contrib.auth.models import User
from unittest.mock import patch, Mock

from apps.projects.models import Project
from apps.projects.services import ProjectService


class ProjectModelTests(TestCase):
    """Test Project model functionality."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    def test_create_project(self):
        """Test creating a new project."""
        project = Project.objects.create(
            name='Test Project',
            owner=self.user,
            description='Test description'
        )

        self.assertEqual(project.name, 'Test Project')
        self.assertEqual(project.owner, self.user)
        self.assertIsNotNone(project.created_at)

    def test_project_str_representation(self):
        """Test string representation of Project."""
        project = Project(name='My Project')
        self.assertEqual(str(project), 'My Project')

    @patch('apps.projects.signals.project_created.send')
    def test_project_creation_signal(self, mock_signal):
        """Test that project creation sends a signal."""
        project = Project.objects.create(
            name='Signal Test',
            owner=self.user
        )

        mock_signal.assert_called_once()
        self.assertEqual(
            mock_signal.call_args[1]['sender'],
            Project
        )


@pytest.mark.django_db
class TestProjectService:
    """Test ProjectService functionality."""

    def test_create_project_with_members(self, user_factory, project_factory):
        """Test creating a project with initial members."""
        owner = user_factory()
        members = user_factory.create_batch(3)

        project = ProjectService.create_project(
            name='Team Project',
            owner=owner,
            members=members
        )

        assert project.owner == owner
        assert project.members.count() == 4  # owner + 3 members

    @patch('apps.projects.services.send_invitation_email')
    def test_invite_member(self, mock_email, project_factory):
        """Test inviting a member to a project."""
        project = project_factory()

        ProjectService.invite_member(
            project=project,
            email='new@example.com'
        )

        mock_email.assert_called_once_with(
            email='new@example.com',
            project=project
        )
```

#### Frontend (React/Next.js)

```typescript
// __tests__/ProjectCard.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { ProjectCard } from '@/components/ProjectCard';
import { projectApi } from '@/services/api/projects';

// Mock API
jest.mock('@/services/api/projects');

const mockProject = {
  id: '1',
  name: 'Test Project',
  description: 'Test description',
  createdAt: '2024-01-01',
  membersCount: 5,
};

const renderWithClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      {ui}
    </QueryClientProvider>
  );
};

describe('ProjectCard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders project information', () => {
    renderWithClient(<ProjectCard project={mockProject} />);

    expect(screen.getByText('Test Project')).toBeInTheDocument();
    expect(screen.getByText('Test description')).toBeInTheDocument();
    expect(screen.getByText(/Members: 5/)).toBeInTheDocument();
  });

  it('handles edit action', async () => {
    const user = userEvent.setup();
    renderWithClient(<ProjectCard project={mockProject} />);

    const editButton = screen.getByRole('button', { name: /edit/i });
    await user.click(editButton);

    await waitFor(() => {
      expect(projectApi.getProject).toHaveBeenCalledWith('1');
    });
  });

  it('calls onUpdate when project is updated', async () => {
    const mockOnUpdate = jest.fn();
    const updatedProject = { ...mockProject, name: 'Updated Project' };

    (projectApi.updateProject as jest.Mock).mockResolvedValue(updatedProject);

    renderWithClient(
      <ProjectCard project={mockProject} onUpdate={mockOnUpdate} />
    );

    // Trigger update flow
    // ... test implementation

    await waitFor(() => {
      expect(mockOnUpdate).toHaveBeenCalledWith(updatedProject);
    });
  });
});
```

### Integration Testing

```javascript
// tests/integration/user-flow.test.js
describe('Complete User Flow', () => {
  let page;

  beforeAll(async () => {
    page = await browser.newPage();
  });

  afterAll(async () => {
    await page.close();
  });

  test('User can register, create project, and invite members', async () => {
    // Registration
    await page.goto('http://localhost:3000/register');
    await page.type('#email', 'newuser@example.com');
    await page.type('#password', 'SecurePass123!');
    await page.click('#submit');

    await page.waitForNavigation();
    expect(page.url()).toContain('/dashboard');

    // Create project
    await page.click('[data-testid="create-project"]');
    await page.type('#project-name', 'Integration Test Project');
    await page.click('#create');

    await page.waitForSelector('[data-project-id]');

    // Invite member
    await page.click('[data-testid="invite-member"]');
    await page.type('#member-email', 'member@example.com');
    await page.click('#send-invite');

    const successMessage = await page.waitForSelector('.success-message');
    expect(await successMessage.textContent()).toContain('Invitation sent');
  });
});
```

## Pull Request Process

### PR Template

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Related Issues
Fixes #(issue)

## Testing
- [ ] Unit tests pass locally
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] New tests added for new functionality

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)
Add screenshots to help explain your changes.
```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline runs tests
   - Code quality checks (linting, formatting)
   - Security scanning
   - Coverage reports

2. **Code Review**
   - At least one maintainer review required
   - Address all feedback constructively
   - Re-request review after making changes

3. **Merge Requirements**
   - All CI checks passing
   - Up-to-date with main branch
   - Approved by maintainer
   - No merge conflicts

## Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, semicolons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvements
- `test`: Adding or correcting tests
- `build`: Changes to build process or dependencies
- `ci`: Changes to CI configuration
- `chore`: Other changes that don't modify src or test files

### Examples
```bash
# Feature
git commit -m "feat(auth): add OAuth2 authentication with Google"

# Bug fix
git commit -m "fix(projects): resolve race condition in project creation"

# Documentation
git commit -m "docs(api): update authentication endpoints documentation"

# With body and footer
git commit -m "feat(subscriptions): add annual billing option

This adds support for annual billing plans with a 20% discount.
Users can now choose between monthly and annual billing cycles.

Closes #123"
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Update API documentation for endpoint changes
- Include examples in documentation
- Keep README files up-to-date

### API Documentation

When adding or modifying API endpoints:

1. Update OpenAPI/Swagger specifications
2. Add request/response examples
3. Document error codes and responses
4. Update SDK documentation if applicable

### User Documentation

For user-facing features:

1. Update user guides
2. Add tooltips and help text
3. Create or update tutorials
4. Add to FAQ if relevant

## Community

### Communication Channels

- **GitHub Discussions**: General discussions and questions
- **Slack**: Real-time chat (invite link in README)
- **Email**: dev@saas-platform.com for development questions
- **Twitter**: @saasplatform for announcements

### Getting Help

- Check existing documentation
- Search closed issues and PRs
- Ask in GitHub Discussions
- Join our Slack community

### Recognition

We value all contributions! Contributors are:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in commit messages
- Featured in our monthly newsletter (major contributions)

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

## Thank You!

Thank you for contributing to the SaaS Web Platform! Your efforts help make this project better for everyone. We appreciate your time and expertise!

If you have any questions or need help, don't hesitate to reach out through any of our communication channels.